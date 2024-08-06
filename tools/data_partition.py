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
from internal.utils.mesh_utils import focus_point_fn
from internal.utils.general_utils import parse
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser

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
                    dataset, 
                    save_dir, 
                    block_dim, 
                    aabb, 
                    num_threshold, 
                    content_threshold, 
                    background_color, 
                    quiet=False, 
                    disable_inblock=False):

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

            xyz = contract_to_unisphere(xyz_org, aabb, ord=torch.inf)

            for block_id in range(block_num):
                block_id_z = block_id // (block_dim[0] * block_dim[1])
                block_id_y = (block_id % (block_dim[0] * block_dim[1])) // block_dim[0]
                block_id_x = (block_id % (block_dim[0] * block_dim[1])) % block_dim[0]

                min_x, max_x = float(block_id_x) / block_dim[0], float(block_id_x + 1) / block_dim[0]
                min_y, max_y = float(block_id_y) / block_dim[1], float(block_id_y + 1) / block_dim[1]
                min_z, max_z = float(block_id_z) / block_dim[2], float(block_id_z + 1) / block_dim[2]

                num_gs, org_min_x, org_max_x, org_min_y, org_max_y, org_min_z, org_max_z = 0, min_x, max_x, min_y, max_y, min_z, max_z

                while num_gs < num_threshold:
                    # TODO: select better threshold
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
                
                print(f"\nStart filtering block {block_id} with {num_gs} gaussians.")
                with open(os.path.join(save_dir, f"block_{block_id}.txt"), "w") as f:
                    for idx in tqdm(range(len(dataset.cameras))):
                        camera = dataset.cameras[idx].to_device("cuda")
                        contract_cam_center = contract_to_unisphere(camera.camera_center, aabb, ord=torch.inf)

                        if (not disable_inblock) and contract_cam_center[0] > org_min_x and contract_cam_center[0] < org_max_x \
                            and contract_cam_center[1] > org_min_y and contract_cam_center[1] < org_max_y \
                            and contract_cam_center[2] > org_min_z and contract_cam_center[2] < org_max_z :
                            f.write(f"{dataset.image_names[idx]}\n")
                            block_image_list[block_id].append(dataset.image_names[idx])
                            continue

                        img_org_gs = renderer(camera, gaussians, bg_color=bg_color)["render"]
                        gaussians.select(block_mask)  # disable gaussians within block
                        img_masked_gs = renderer(camera, gaussians, bg_color=bg_color)["render"]
                        gaussians._opacity = gaussians._opacity_origin  # recover opacity

                        loss = 1.0 - ssim(img_masked_gs, img_org_gs)
                        if loss > content_threshold:
                            f.write(f"{dataset.image_names[idx]}\n")
                            block_image_list[block_id].append(dataset.image_names[idx])
        
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
        args.block_dim = config.data.params.colmap_block.block_dim
        args.aabb = config.data.params.colmap_block.aabb
        args.num_threshold = config.data.params.colmap_block.num_threshold
        args.content_threshold = config.data.params.colmap_block.content_threshold

        if args.save_dir is None:
            save_dir = config.data.params.colmap_block.image_list
        else:
            save_dir = args.save_dir
        
        if 'point_cloud' in config.model.init_from:
            args.model_path = config.model.init_from.split("/point_cloud/")[0]
        elif 'checkpoints' in config.model.init_from:
            args.model_path = config.model.init_from.split("/checkpoints/")[0]
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    model, renderer = GaussianModelLoader.search_and_load(
        args.model_path,
        sh_degree=3,
        device="cuda",
    )
    print("Gaussian count: {}".format(model.get_xyz.shape[0]))

    config_path = os.path.join(args.model_path, "config.yaml")
    with open(config_path, 'r') as f:
        config = parse(yaml.load(f, Loader=yaml.FullLoader))
    
    # TODO: support other data parser
    dataparser_outputs = ColmapBlockDataParser(
        os.path.expanduser(config.data.path),
        os.path.abspath(""),
        global_rank=0,
        params=config.data.params.colmap_block,
    ).get_outputs()
    
    
    # assert save_dir contains no files and avoid duplicated partitioning
    assert len(os.listdir(save_dir)) == 0, f"{save_dir} already contains partition files!"

    block_filtering(model, renderer, 
                    dataparser_outputs.train_set, 
                    save_dir, args.block_dim, args.aabb, 
                    args.num_threshold, args.content_threshold,
                    config.model.background_color, 
                    disable_inblock=False)

    # All done
    print("Partition complete.")