import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.ssim import ssim
from internal.utils.blocking import contract_to_unisphere
from internal.utils.general_utils import focus_point_fn
from internal.utils.general_utils import parse
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthColmapDataParser
from internal.renderers.vanilla_trim_renderer import VanillaTrimRenderer

def contract(x):
    mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
    out = torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag)) # [-inf, inf] is at [-2, 2]
    out = out / 4 + 0.5  # [-2, 2] is at [0, 1]
    return out

def uncontract(y):
    y = y * 4 - 2  # [0, 1] is at [-2, 2]
    mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
    return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

def block_filtering(gaussians, 
                    renderer, 
                    ckpt_path,
                    dataset, 
                    save_dir, 
                    block_dim, 
                    aabb, 
                    num_threshold, 
                    content_threshold, 
                    background_color, 
                    quiet=False, 
                    flatten_gs=False,
                    disable_inblock=False):

        checkpoint = torch.load(ckpt_path)
        max_radii2D = checkpoint["gaussian_model_extra_state_dict"]["max_radii2D"].clone()
        xyz_gradient_accum = checkpoint["gaussian_model_extra_state_dict"]["xyz_gradient_accum"].clone()
        denom = checkpoint["gaussian_model_extra_state_dict"]["denom"].clone()
        device = denom.device

        xyz_org = gaussians.get_xyz
        block_num = block_dim[0] * block_dim[1] * block_dim[2]
        bg_color=torch.tensor(background_color, dtype=torch.float, device="cuda")

        if aabb is None:
            torch.cuda.empty_cache()
            c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in dataset.cameras])
            poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
            center = (focus_point_fn(poses))
            radius = torch.tensor(np.median(np.abs(c2ws[:,:3,3] - center), axis=0), device=xyz_org.device)
            center = torch.from_numpy(center).float().to(xyz_org.device)
            if radius.min() / radius.max() < 0.02:
                # If the radius is too small, we don't contract in this dimension
                radius[torch.argmin(radius)] = 0.5 * (xyz_org[:, torch.argmin(radius)].max() - xyz_org[:, torch.argmin(radius)].min())
            aabb = torch.zeros(6, device=xyz_org.device)
            aabb[:3] = center - radius
            aabb[3:] = center + radius
        else:
            assert len(aabb) == 6, "Unknown aabb format!"
            aabb = torch.tensor(aabb, dtype=torch.float32, device=xyz_org.device)
        
        print(f"Block number: {block_num}, Gaussian number threshold: {num_threshold}")

        block_image_list = {i: [] for i in range(block_num)}
        
        with torch.no_grad():
            # find even quantile of gaussians along each axis according to block_dim
            x_quantile = torch.quantile(xyz_org[:, 0], torch.linspace(0, 1, block_dim[0] + 1).to(device))
            y_quantile = torch.quantile(xyz_org[:, 1], torch.linspace(0, 1, block_dim[1] + 1).to(device))
            z_quantile = torch.quantile(xyz_org[:, 2], torch.linspace(0, 1, block_dim[2] + 1).to(device))
            x_quantile[0] -= 1e6
            y_quantile[0] -= 1e6
            z_quantile[0] -= 1e6
            x_quantile[-1] += 1e6
            y_quantile[-1] += 1e6
            z_quantile[-1] += 1e6
            # save quantiles for future use
            torch.save({"x": x_quantile, "y": y_quantile, "z": z_quantile}, os.path.join(save_dir, "quantiles.pt"))

            for block_id in range(block_num):
                block_id_z = block_id // (block_dim[0] * block_dim[1])
                block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
                block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

                min_x, max_x = x_quantile[block_id_x].item(), x_quantile[block_id_x + 1].item()
                min_y, max_y = y_quantile[block_id_y].item(), y_quantile[block_id_y + 1].item()
                min_z, max_z = z_quantile[block_id_z].item(), z_quantile[block_id_z + 1].item()

                block_mask = (xyz_org[:, 0] >= min_x) & (xyz_org[:, 0] < max_x)  \
                            & (xyz_org[:, 1] >= min_y) & (xyz_org[:, 1] < max_y) \
                            & (xyz_org[:, 2] >= min_z) & (xyz_org[:, 2] < max_z)
                
                block_output_mask = block_mask.clone()

                print(f"\nStart filtering block {block_id} with {block_mask.sum()} gaussians.")
                with open(os.path.join(save_dir, f"block_{block_id}.txt"), "w") as f:
                    for idx in tqdm(range(len(dataset.cameras))):
                        camera = dataset.cameras[idx].to_device("cuda")

                        output = renderer(camera, gaussians, bg_color=bg_color)

                        if (not disable_inblock) and camera.camera_center[0] > min_x and camera.camera_center[0] < max_x \
                            and camera.camera_center[1] > min_y and camera.camera_center[1] < max_y \
                            and camera.camera_center[2] > min_z and camera.camera_center[2] < max_z :
                            f.write(f"{dataset.image_names[idx]}\n")
                            block_image_list[block_id].append(dataset.image_names[idx])
                        else:
                            img_org_gs = output["render"]
                            gaussians.select(block_mask)  # disable gaussians within block
                            img_masked_gs = renderer(camera, gaussians, bg_color=bg_color)["render"]
                            gaussians._opacity = gaussians._opacity_origin  # recover opacity

                            loss = 1.0 - ssim(img_masked_gs, img_org_gs)
                            if loss > content_threshold:
                                f.write(f"{dataset.image_names[idx]}\n")
                                block_image_list[block_id].append(dataset.image_names[idx])
                        
                        if dataset.image_names[idx] in block_image_list[block_id]:
                            block_output_mask = block_output_mask | output['visibility_filter']
                    
                    # save filtered gaussians
                    block_output_mask = block_output_mask
                    gaussians_params = gaussians.to_parameter_structure(device)
                    if flatten_gs:
                        gaussians_params.scales = gaussians_params.scales[:, 1:]

                    checkpoint["state_dict"]["gaussian_model._xyz"] = gaussians_params.xyz[block_output_mask]
                    checkpoint["state_dict"]["gaussian_model._opacity"] = gaussians_params.opacities[block_output_mask]
                    checkpoint["state_dict"]["gaussian_model._features_dc"] = gaussians_params.features_dc[block_output_mask]
                    checkpoint["state_dict"]["gaussian_model._features_rest"] = gaussians_params.features_rest[block_output_mask]
                    checkpoint["state_dict"]["gaussian_model._scaling"] = gaussians_params.scales[block_output_mask]
                    checkpoint["state_dict"]["gaussian_model._rotation"] = gaussians_params.rotations[block_output_mask]
                    checkpoint["state_dict"]["gaussian_model._features_extra"] = gaussians_params.real_features_extra[block_output_mask]
                    checkpoint["gaussian_model_extra_state_dict"]["max_radii2D"] = max_radii2D[block_output_mask].clone()
                    checkpoint["gaussian_model_extra_state_dict"]["xyz_gradient_accum"] = xyz_gradient_accum[block_output_mask].clone()
                    checkpoint["gaussian_model_extra_state_dict"]["denom"] = denom[block_output_mask].clone()
                    # torch.save(checkpoint, ckpt_path.replace(".ckpt", f"_block_{block_id}.ckpt"))
        
        if not quiet:
            for block_id in range(block_num):
                print(f"Block {block_id} / {block_num} has {len(block_image_list[block_id])} cameras.")
                    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config_path', type=str, help='path of finetuned model', default=None)
    parser.add_argument('--model_path', type=str, help='path of coarse global model')
    parser.add_argument("--block_dim", type=int, nargs="+", default=None)
    parser.add_argument("--aabb", type=float, nargs="+", default=None)
    parser.add_argument("--num_threshold", type=int, default=25000)
    parser.add_argument("--content_threshold", type=float, default=0.08)
    parser.add_argument("--save_dir", type=str, help="directory to save partition", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_inblock", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        # parameters in config file will overwrite command line arguments
        print(f"Loading parameters according to config file {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = parse(yaml.load(f, Loader=yaml.FullLoader))
            if config.data.type == "estimated_depth_colmap_block":
                params = config.data.params.estimated_depth_colmap_block
            else:
                params = config.data.params.colmap_block
        args.block_dim = params.block_dim
        args.aabb = params.aabb
        args.num_threshold = params.num_threshold
        args.content_threshold = params.content_threshold

        if args.save_dir is None:
            save_dir = params.image_list
        else:
            save_dir = args.save_dir
        
        ckpt_path = config.model.init_from
        if 'point_cloud' in config.model.init_from:
            args.model_path = config.model.init_from.split("/point_cloud/")[0]
        elif 'checkpoints' in config.model.init_from:
            args.model_path = config.model.init_from.split("/checkpoints/")[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model, renderer, _ = GaussianModelLoader.initialize_simplified_model_from_checkpoint(ckpt_path, device="cuda")
    print("Gaussian count: {}".format(model.get_xyz.shape[0]))

    if isinstance(renderer, VanillaTrimRenderer):
        model._scaling = torch.cat((torch.ones_like(model._scaling[:, :1]) * 1e-8, model._scaling[:, [-2, -1]]), dim=1)
        flatten_gs = True
    else:
        flatten_gs = False

    config_path = os.path.join(args.model_path, "config.yaml")
    with open(config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))
    
    # TODO: support other data parser
    if config.data.type == "estimated_depth_colmap_block":
        dataparser_outputs = EstimatedDepthColmapDataParser(
            os.path.expanduser(config.data.path),
            os.path.abspath(""),
            global_rank=0,
            params=config.data.params.estimated_depth_colmap_block,
        ).get_outputs()
    else:
        dataparser_outputs = ColmapBlockDataParser(
            os.path.expanduser(config.data.path),
            os.path.abspath(""),
            global_rank=0,
            params=config.data.params.colmap_block,
        ).get_outputs()
    
    
    # assert save_dir contains no files and avoid duplicated partitioning
    # assert len(os.listdir(save_dir)) == 0, f"{save_dir} already contains partition files!"

    block_filtering(model, renderer, ckpt_path,
                    dataparser_outputs.train_set, 
                    save_dir, args.block_dim, args.aabb, 
                    args.num_threshold, args.content_threshold,
                    config.model.background_color, 
                    flatten_gs=flatten_gs, disable_inblock=False)

    # All done
    print("Partition complete.")