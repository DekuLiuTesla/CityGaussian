import os
import sys
import yaml
import torch
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from internal.utils.general_utils import parse
from internal.utils.render_utils import generate_path
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config_path', type=str, help='path of config', default=None)
    parser.add_argument('--mesh_path', type=str, help='path of reconstructed mesh')
    parser.add_argument("--n_fames", type=int, help="number of frames", default=240)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--disable_inblock", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        # parameters in config file will overwrite command line arguments
        print(f"Loading parameters according to config file {args.config_path}")
        with open(args.config_path, 'r') as f:
            config = parse(yaml.load(f, Loader=yaml.FullLoader))
    
    # TODO: support other data parser
    dataparser_outputs = ColmapBlockDataParser(
        os.path.expanduser(config.data.path),
        os.path.abspath(""),
        global_rank=0,
        params=config.data.params.colmap_block,
    ).get_outputs()
    
    traj_dir = os.path.join(args.mesh_path, 'traj')
    os.makedirs(traj_dir, exist_ok=True)
    cam_traj = generate_path(dataparser_outputs.train_set.cameras, traj_dir, n_frames=args.n_fames)

    print("Camera trajectory generated")