import os
import sys
import yaml
import torch
import torch_scatter
import imageio
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.general_utils import parse
from internal.utils.render_utils import generate_path, record_path
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser
from internal.dataparsers.estimated_depth_colmap_block_dataparser import EstimatedDepthColmapDataParser
from internal.renderers.vanilla_trim_renderer import VanillaTrimRenderer

def voxel_filtering_no_gt(voxel_size, xy_range, target_xyz, std_ratio=2.0):
    assert len(xy_range) == 4, "Unrecognized xy_range format"
    with torch.no_grad():

        voxel_index = torch.div(torch.tensor(target_xyz[:, :2]).float() - xy_range[None, :2], voxel_size[None, :], rounding_mode='floor')
        voxel_coords = voxel_index * voxel_size[None, :] + xy_range[None, :2] + voxel_size[None, :] / 2

        new_coors, unq_inv, unq_cnt = torch.unique(voxel_coords, return_inverse=True, return_counts=True, dim=0)
        feat_mean = torch_scatter.scatter(target_xyz[:, 2], unq_inv, dim=0, reduce='mean')
        feat_std = torch_scatter.scatter_std(target_xyz[:, 2], unq_inv, dim=0)

        mask = target_xyz[:, 2] > feat_mean[unq_inv] + std_ratio * feat_std[unq_inv]

    return mask

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config_path', type=str, help='path of config', default=None)
    parser.add_argument('--data_path', type=str, help='path of data', default=None)
    parser.add_argument('--ckpt_path', type=str, help='path of data', default=None)
    parser.add_argument("--n_frames", type=int, help="number of frames", default=240)
    parser.add_argument("--scale_percentile", type=int, help="trajectory radius percentile", default=99)
    parser.add_argument("--pitch", type=float, help="pitch in degree, 0 means no pitch changes", default=None)
    parser.add_argument("--x_shift", type=float, help="x-axis shift of ellipse center, 0 means no pitch changes", default=0)
    parser.add_argument("--y_shift", type=float, help="y-axis shift of ellipse center, 0 means no pitch changes", default=0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter", action="store_true")
    parser.add_argument("--vox_grid", type=int, help="number of voxelization grid", default=25)
    parser.add_argument("--std_ratio", type=float, help="used to control filtering threshold", default=2.0)
    parser.add_argument("--train", action="store_true", help="whether to use train set as trajectories")
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        # parameters in config file will overwrite command line arguments
        print(f"Loading parameters according to config file {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = parse(yaml.load(f, Loader=yaml.FullLoader))
            if args.data_path is not None:
                config.data.path = args.data_path
    
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

    if args.train:
        cameras = dataparser_outputs.train_set.cameras
    else:
        cameras = dataparser_outputs.test_set.cameras

    model, renderer, _ = GaussianModelLoader.initialize_simplified_model_from_checkpoint(args.ckpt_path, device="cuda")
    bg_color=torch.tensor(config.model.background_color, dtype=torch.float, device="cuda")
    print("Gaussian count: {}".format(model.get_xyz.shape[0]))

    traj_dir = os.path.join(config.data.path, 'traj_ellipse')
    os.makedirs(traj_dir, exist_ok=True)

    if not args.filter:
        cam_traj = generate_path(cameras, traj_dir, n_frames=args.n_frames, pitch=args.pitch, shift=args.shift, scale_percentile=args.scale_percentile)
    else:
        cam_traj, colmap_to_world_transform, pose_recenter = generate_path(cameras, traj_dir, n_frames=args.n_frames, pitch=args.pitch, 
                                                                           shift=[args.x_shift, args.y_shift], filter=True, 
                                                                           scale_percentile=args.scale_percentile)
        xyz_homo = torch.cat((model.get_xyz, torch.zeros(model.get_xyz.shape[0], 1, device="cuda")), dim=-1)
        transformed_xyz = xyz_homo @ torch.tensor(colmap_to_world_transform, device="cuda", dtype=xyz_homo.dtype).T
        x_min, x_max = pose_recenter[:, 0, -1].min(), pose_recenter[:, 0, -1].max()
        y_min, y_max = pose_recenter[:, 1, -1].min(), pose_recenter[:, 1, -1].max()
        voxel_size = torch.tensor([(x_max - x_min) / args.vox_grid, (y_max - y_min) / args.vox_grid], device="cuda")
        xy_range = torch.tensor([x_min, y_min, x_max, y_max], device="cuda")
        vox_mask = voxel_filtering_no_gt(voxel_size, xy_range, transformed_xyz, args.std_ratio).bool().cpu().numpy()
        model.select(vox_mask)
    
    print(f"Camera trajectory saved to {traj_dir}. Start rendering...")

    if isinstance(renderer, VanillaTrimRenderer):
        model._scaling = torch.cat((torch.ones_like(model._scaling[:, :1]) * 1e-8, model._scaling[:, [-2, -1]]), dim=1)
        flatten_gs = True
    else:
        flatten_gs = False

    os.makedirs('./videos', exist_ok=True)
    video_path = os.path.join('./videos', f"{args.ckpt_path.split('/')[-3]}_video.mp4")
    video = imageio.get_writer(video_path, fps=30)
    for t in tqdm(range(len(cam_traj))):
        cam = cam_traj[t]
        cam.height = torch.tensor(cam.height, device=cam.R.device)
        cam.width = torch.tensor(cam.width, device=cam.R.device)
        img = renderer(cam, model, bg_color)['render']
        img = (img * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        video.append_data(img)
    video.close()
    print(f"Video saved to {video_path}.")
