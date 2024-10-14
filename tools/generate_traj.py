import os
import sys
import yaml
import torch
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

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config_path', type=str, help='path of config', default=None)
    parser.add_argument('--data_path', type=str, help='path of data', default=None)
    parser.add_argument("--n_fames", type=int, help="number of frames", default=240)
    parser.add_argument("--scale_percentile", type=int, help="trajectory radius percentile", default=99)
    parser.add_argument("--pitch", type=float, help="pitch in degree, 0 means no pitch changes", default=None)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--ellipse", action="store_true", help="whether to generate new trajectories of ellipse shape")
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
    
    traj_dir = os.path.join(config.data.path, 'traj')
    if args.ellipse:
        traj_dir = traj_dir + '_ellipse'
        os.makedirs(traj_dir, exist_ok=True)
        cam_traj = generate_path(cameras, traj_dir, n_frames=args.n_fames, scale_percentile=args.scale_percentile)
    else:
        os.makedirs(traj_dir, exist_ok=True)
        cam_traj = record_path(cameras, traj_dir)

    print(f"Camera trajectory saved to {traj_dir}")