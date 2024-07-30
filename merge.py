#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import yaml
import os
import sys

import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as tf
from PIL import Image
from tqdm import tqdm
from os import makedirs
from scene import Scene
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams, GroupParams
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import parse_cfg

def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names

def blockMerge(lp, iteration):
    out_dir = lp.model_path
    merged_gaussians = GaussianModel(lp.sh_degree)
    num_blocks = lp.block_dim[0] * lp.block_dim[1] * lp.block_dim[2]
    with torch.no_grad():
        for idx in range(num_blocks):
            gaussians = GaussianModel(lp.sh_degree)
            try:
                gaussians.load_ply(os.path.join(out_dir, f"cells/cell{idx}", "point_cloud_blocks", "scale_1.0",
                                                "iteration_" + str(iteration),
                                                "point_cloud.ply"))
                num_iter = iteration
            except:
                gaussians.load_ply(os.path.join(out_dir, f"cells/cell{idx}", "point_cloud_blocks", "scale_1.0",
                                                "iteration_" + str(1),
                                                "point_cloud.ply"))
                num_iter = 1
            
            if len(merged_gaussians._xyz) == 0:
                merged_gaussians._xyz = gaussians.get_xyz
                merged_gaussians._features_dc = gaussians._features_dc
                merged_gaussians._features_rest = gaussians._features_rest
                merged_gaussians._scaling = gaussians._scaling
                merged_gaussians._rotation = gaussians._rotation
                merged_gaussians._opacity = gaussians._opacity
                merged_gaussians.max_radii2D = gaussians.max_radii2D
            else:
                merged_gaussians._xyz = torch.cat([merged_gaussians._xyz, gaussians.get_xyz], dim=0)
                merged_gaussians._features_dc = torch.cat([merged_gaussians._features_dc, gaussians._features_dc], dim=0)
                merged_gaussians._features_rest = torch.cat([merged_gaussians._features_rest, gaussians._features_rest], dim=0)
                merged_gaussians._scaling = torch.cat([merged_gaussians._scaling, gaussians._scaling], dim=0)
                merged_gaussians._rotation = torch.cat([merged_gaussians._rotation, gaussians._rotation], dim=0)
                merged_gaussians._opacity = torch.cat([merged_gaussians._opacity, gaussians._opacity], dim=0)
                merged_gaussians.max_radii2D = torch.cat([merged_gaussians.max_radii2D, gaussians.max_radii2D], dim=0)
            
            print(f"Merged {len(gaussians.get_xyz)} points from block {idx} from iteration {num_iter}.")
    
    save_path = os.path.join(out_dir, "point_cloud", "iteration_" + str(iteration), "point_cloud.ply")
    print(f"Saving merged {len(merged_gaussians.get_xyz)} point cloud to {save_path}")
    merged_gaussians.save_ply(save_path)
    print('Done')


if __name__ == "__main__":
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--config', type=str, help='train config file path')
    parser.add_argument('--model_path', type=str, help='model path of fused model')
    parser.add_argument("--iteration", default=30_000, type=int)
    args = parser.parse_args(sys.argv[1:])
    if args.model_path is None:
        args.model_path = os.path.join('output', os.path.basename(args.config).split('.')[0])

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)

    blockMerge(lp, args.iteration)
