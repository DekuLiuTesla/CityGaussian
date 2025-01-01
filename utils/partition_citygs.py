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

def contract(x):
    mag = torch.linalg.norm(x, ord=2, dim=-1)[..., None]
    out = torch.where(mag < 1, x, (2 - (1 / mag)) * (x / mag)) # [-inf, inf] is at [-2, 2]
    out = out / 4 + 0.5  # [-2, 2] is at [0, 1]
    return out

def uncontract(y):
    y = y * 4 - 2  # [0, 1] is at [-2, 2]
    mag = torch.linalg.norm(y, ord=2, dim=-1)[..., None]
    return torch.where(mag < 1, y, (1 / (2-mag) * (y/mag)))

def auto_extend(xyz, min_x, max_x, min_y, max_y, min_z, max_z, num_threshold=None):
    num_gs = 0

    if num_threshold is not None:
        while num_gs < num_threshold:
            block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                        & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                        & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)
            num_gs = block_mask.sum()
            min_x -= 0.01
            max_x += 0.01
            min_y -= 0.01
            max_y += 0.01
            min_z -= 0.01
            max_z += 0.01
    
    block_mask = (xyz[:, 0] >= min_x) & (xyz[:, 0] < max_x)  \
                & (xyz[:, 1] >= min_y) & (xyz[:, 1] < max_y) \
                & (xyz[:, 2] >= min_z) & (xyz[:, 2] < max_z)
    
    return min_x, max_x, min_y, max_y, min_z, max_z, block_mask

def block_filtering(gaussians, 
                    renderer, 
                    dataset,
                    save_dir,
                    args, 
                    background_color):

        xyz_org = gaussians.get_xyz
        opacity_org = torch.clone(gaussians.get_opacities())
        block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]
        bg_color=torch.tensor(background_color, dtype=torch.float, device="cuda")

        if args.contract:
            if args.aabb is None:
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
                assert len(args.aabb) == 6, "Unknown aabb format!"
                aabb = torch.tensor(args.aabb, dtype=torch.float32, device=xyz_org.device)
            
            if args.num_threshold is None:
                args.num_threshold = xyz_org.shape[0] // block_num
            
            xyz = contract_to_unisphere(xyz_org, aabb, ord=torch.inf)
            x_quantile = torch.arange(0, 1+1/args.block_dim[0], 1/args.block_dim[0], device=xyz.device)
            y_quantile = torch.arange(0, 1+1/args.block_dim[1], 1/args.block_dim[1], device=xyz.device)
            z_quantile = torch.arange(0, 1+1/args.block_dim[2], 1/args.block_dim[2], device=xyz.device)
        
        else:
            xyz = xyz_org
            
            # find even quantile of gaussians along each axis according to block_dim
            x_quantile = torch.quantile(xyz_org[:, 0], torch.linspace(0, 1, args.block_dim[0] + 1).to(device))
            y_quantile = torch.quantile(xyz_org[:, 1], torch.linspace(0, 1, args.block_dim[1] + 1).to(device))
            z_quantile = torch.quantile(xyz_org[:, 2], torch.linspace(0, 1, args.block_dim[2] + 1).to(device))
            x_quantile[0] -= 1e6
            y_quantile[0] -= 1e6
            z_quantile[0] -= 1e6
            x_quantile[-1] += 1e6
            y_quantile[-1] += 1e6
            z_quantile[-1] += 1e6
            # save quantiles for future use
            torch.save({"x": x_quantile, "y": y_quantile, "z": z_quantile}, os.path.join(save_dir, "quantiles.pt"))

        block_image_list = {i: [] for i in range(block_num)}
        
        with torch.no_grad():

            for block_id in range(block_num):
                block_id_z = block_id // (args.block_dim[0] * args.block_dim[1])
                block_id_y = (block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
                block_id_x = (block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

                min_x, max_x = x_quantile[block_id_x].item(), x_quantile[block_id_x + 1].item()
                min_y, max_y = y_quantile[block_id_y].item(), y_quantile[block_id_y + 1].item()
                min_z, max_z = z_quantile[block_id_z].item(), z_quantile[block_id_z + 1].item()

                min_x, max_x, min_y, max_y, min_z, max_z, block_mask = \
                    auto_extend(xyz, min_x, max_x, min_y, max_y, min_z, max_z, args.num_threshold)

                print(f"\nStart filtering block {block_id} with {block_mask.sum()} gaussians.")
                with open(os.path.join(save_dir, f"block_{block_id}.txt"), "w") as f:
                    for idx in tqdm(range(len(dataset.cameras))):
                        camera = dataset.cameras[idx].to_device("cuda")

                        camera_center = camera.camera_center if not args.contract else contract_to_unisphere(camera.camera_center, aabb, ord=torch.inf)

                        output = renderer(camera, gaussians, bg_color=bg_color)

                        if camera_center[0] > min_x and camera_center[0] < max_x \
                            and camera_center[1] > min_y and camera_center[1] < max_y \
                            and camera_center[2] > min_z and camera_center[2] < max_z :
                            f.write(f"{dataset.image_names[idx]}\n")
                            block_image_list[block_id].append(dataset.image_names[idx])
                        else:
                            img_org_gs = output["render"]
                            gaussians.opacities[block_mask] = 0.0
                            img_masked_gs = renderer(camera, gaussians, bg_color=bg_color)["render"]
                            gaussians.opacities = opacity_org.clone()  # recover opacity

                            loss = 1.0 - ssim(img_masked_gs, img_org_gs)
                            if loss > args.content_threshold:
                                f.write(f"{dataset.image_names[idx]}\n")
                                block_image_list[block_id].append(dataset.image_names[idx])
        
        if not args.quiet:
            for block_id in range(block_num):
                print(f"Block {block_id} / {block_num} has {len(block_image_list[block_id])} cameras.")
                    

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_path", type=str, help="path of finetuned model", default=None)
    parser.add_argument("--model_path", type=str, help="path of coarse global model")
    parser.add_argument("--contract", action="store_true", help="whether partition in contracted space, suitable for irregular distribution")
    parser.add_argument("--save_dir", type=str, help="directory to save partition", default=None)
    parser.add_argument("--content_threshold", type=float, default=0.08)
    parser.add_argument("--block_dim", type=int, nargs="+", default=None)
    parser.add_argument("--aabb", type=float, nargs="+", default=None)
    parser.add_argument("--num_threshold", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        # parameters in config file will overwrite command line arguments
        print(f"Loading parameters according to config file {args.config_path}")
        with open(args.config_path, "r") as f:
            config = parse(yaml.load(f, Loader=yaml.FullLoader))
            params = config.data.parser.init_args
        args.block_dim = params.block_dim
        args.content_threshold = params.content_threshold if hasattr(params, "content_threshold") else args.content
        if args.contract:
            args.aabb = params.aabb if hasattr(params, "aabb") else args.aabb
            args.num_threshold = params.num_threshold if hasattr(params, "num_threshold") else args.num_threshold

        if args.save_dir is None:
            save_dir = params.image_list
        else:
            save_dir = args.save_dir
        
        ckpt_path = config.model.initialize_from
        if "point_cloud" in config.model.initialize_from:
            args.model_path = config.model.initialize_from.split("/point_cloud/")[0]
        elif "checkpoints" in config.model.initialize_from:
            args.model_path = config.model.initialize_from.split("/checkpoints/")[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # initialize model
    device = torch.device("cuda")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bkgd_color = ckpt["hyper_parameters"]["background_color"]
    model = GaussianModelLoader.initialize_model_from_checkpoint(
        ckpt,
        device=device,
    )
    model.freeze()
    model.pre_activate_all_properties()
    # initialize renderer
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
        ckpt,
        stage="validate",
        device=device,
    )
    print("Gaussian count: {}".format(model.get_xyz.shape[0]))

    # initialize dataset
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
    dataparser_outputs = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()
    
    # assert save_dir contains no files and avoid duplicated partitioning
    if not args.force:
        assert len(os.listdir(save_dir)) == 0, f"{save_dir} already contains partition files!, use --force to overwrite"

    block_filtering(model, renderer, dataparser_outputs.train_set, 
                    save_dir, args, bkgd_color)

    # All done
    print("Partition complete.")