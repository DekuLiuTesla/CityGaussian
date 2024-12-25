import os
import sys
import json
import yaml
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser, Namespace
from transforms3d.quaternions import mat2quat
from scene import LargeScene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render
from utils.general_utils import safe_state, parse_cfg
from utils.large_utils import contract_to_unisphere, get_default_aabb
from utils.loss_utils import ssim
from utils.camera_utils import loadCam_woImage
from arguments import GroupParams

def block_partitioning(cameras, gaussians, args, pp, scale=1.0, quiet=False, disable_inblock=False, simple_selection=False):

        xyz_org = gaussians.get_xyz
        num_threshold = args.num_threshold
        block_num = args.block_dim[0] * args.block_dim[1] * args.block_dim[2]

        if args.aabb is None:
            torch.cuda.empty_cache()
            args.aabb = get_default_aabb(args, cameras, xyz_org, scale)
            config_name = os.path.splitext(os.path.basename(args.config))[0]
            np.save(os.path.join(args.source_path, "data_partitions", f"{config_name}_aabb.npy"), np.array(args.aabb.detach().cpu()))
        else:
            assert len(args.aabb) == 6, "Unknown args.aabb format!"
            args.aabb = torch.tensor(args.aabb, dtype=torch.float32, device=xyz_org.device)
        
        print(f"Block number: {block_num}, Gaussian number threshold: {num_threshold}")

        camera_mask = torch.zeros((len(cameras), block_num), dtype=torch.bool, device=xyz_org.device)
        
        with torch.no_grad():

            for block_id in range(block_num):
                block_id_z = block_id // (args.block_dim[0] * args.block_dim[1])
                block_id_y = (block_id % (args.block_dim[0] * args.block_dim[1])) // args.block_dim[0]
                block_id_x = (block_id % (args.block_dim[0] * args.block_dim[1])) % args.block_dim[0]

                xyz = contract_to_unisphere(xyz_org, args.aabb, ord=torch.inf)
                min_x, max_x = float(block_id_x) / args.block_dim[0], float(block_id_x + 1) / args.block_dim[0]
                min_y, max_y = float(block_id_y) / args.block_dim[1], float(block_id_y + 1) / args.block_dim[1]
                min_z, max_z = float(block_id_z) / args.block_dim[2], float(block_id_z + 1) / args.block_dim[2]

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
                
                block_mask = ~block_mask
                sh_degree = gaussians.max_sh_degree
                masked_gaussians = GaussianModel(sh_degree)
                masked_gaussians._xyz = xyz_org[block_mask]
                masked_gaussians._scaling = gaussians._scaling[block_mask]
                masked_gaussians._rotation = gaussians._rotation[block_mask]
                masked_gaussians._features_dc = gaussians._features_dc[block_mask]
                masked_gaussians._features_rest = gaussians._features_rest[block_mask]
                masked_gaussians._opacity = gaussians._opacity[block_mask]
                masked_gaussians.max_radii2D = gaussians.max_radii2D[block_mask]

                for idx in tqdm(range(len(cameras)), desc=f"Block {block_id} / {block_num}"):
                    bg_color = [1,1,1] if args.white_background else [0, 0, 0]
                    background = torch.tensor(bg_color, dtype=torch.float32, device=xyz_org.device)
                    c = cameras[idx]
                    viewpoint_cam = loadCam_woImage(args, idx, c, scale)
                    contract_cam_center = contract_to_unisphere(viewpoint_cam.camera_center, args.aabb, ord=torch.inf)

                    if simple_selection > 1.0:
                        # enlarge the box to 1.5x
                        rate = (simple_selection - 1.0) / 2
                        min_x -= rate * (org_max_x - org_min_x)
                        max_x += rate * (org_max_x - org_min_x)
                        min_y -= rate * (org_max_y - org_min_y)
                        max_y += rate * (org_max_y - org_min_y)
                        min_z -= rate * (org_max_z - org_min_z)
                        max_z += rate * (org_max_z - org_min_z)
                        if contract_cam_center[0] > min_x and contract_cam_center[0] < max_x \
                            and contract_cam_center[1] > min_y and contract_cam_center[1] < max_y \
                            and contract_cam_center[2] > min_z and contract_cam_center[2] < max_z :
                            camera_mask[idx, block_id] = True
                        continue

                    if (not disable_inblock) and contract_cam_center[0] > org_min_x and contract_cam_center[0] < org_max_x \
                        and contract_cam_center[1] > org_min_y and contract_cam_center[1] < org_max_y \
                        and contract_cam_center[2] > org_min_z and contract_cam_center[2] < org_max_z :
                        camera_mask[idx, block_id] = True
                        continue

                    render_pkg_block = render(viewpoint_cam, gaussians, pp, background)

                    org_image_block = render_pkg_block["render"]
                    render_pkg_block = render(viewpoint_cam, masked_gaussians, pp, background)
                    image_block = render_pkg_block["render"]
                    loss = 1.0 - ssim(image_block, org_image_block)
                    if loss > args.ssim_threshold:
                        camera_mask[idx, block_id] = True
        
        if not quiet:
            for block_id in range(block_num):
                print(f"Block {block_id} / {block_num} has {camera_mask[:, block_id].sum()} cameras.")
                    
        return camera_mask


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true')
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_inblock", action="store_true")
    parser.add_argument("--simple_selection", type=float, default=0)
    args = parser.parse_args(sys.argv[1:])
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    config_name = os.path.splitext(os.path.basename(lp.config))[0]
    if not lp.model_path:
        # time_stamp = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
        lp.model_path = os.path.join("./output/", config_name)
    
    print("Output folder: {}".format(lp.model_path))
    os.makedirs(lp.model_path, exist_ok = True)
    with open(os.path.join(lp.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(lp))))

    modules = __import__('scene')
    model_config = lp.model_config
    gaussians = getattr(modules, model_config['name'])(lp.sh_degree, **model_config['kwargs'])
    scene = LargeScene(lp, gaussians, shuffle=False)
    if not os.path.exists(os.path.join(lp.source_path, "data_partitions")):
        os.makedirs(os.path.join(lp.source_path, "data_partitions"))
    camera_mask = block_partitioning(scene.getTrainCameras(), gaussians, lp, pp, 1.0, args.quiet, args.disable_inblock, args.simple_selection)
    camera_mask = camera_mask.cpu().numpy()
    np.save(os.path.join(lp.source_path, "data_partitions", f"{config_name}.npy"), camera_mask)

    # All done
    print("\Partition complete.")