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

import os
import sys
import yaml
import torch
import torchvision
import wandb
import time
import inspect
import numpy as np
import pynvml
from tqdm import tqdm
from arguments import GroupParams
from scene import LargeScene
from scene.datasets import GSDataset
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.data import DataLoader
from utils.camera_utils import loadCamTA1

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, pitch, heights):
    avg_render_time = 0
    max_render_time = 0
    avg_memory = 0
    max_memory = 0

    render_path = os.path.join(model_path, name, "ours_TA1", "renders")
    gts_path = os.path.join(model_path, name, "ours_TA1", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    view = views[len(views) // 2]
    height_arr = np.arange(heights[0], heights[1], (heights[1] - heights[0]) / heights[2])

    for idx in tqdm(range(heights[-1]), desc="Rendering progress"):
        
        viewpoint_cam = loadCamTA1(lp, idx, view, 1.0, pitch, height_arr[idx])

        # gpu_tracker.track() 
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        render_pkg = render(viewpoint_cam, gaussians, pipeline, background)
        torch.cuda.synchronize()
        end = time.time()
        avg_render_time += end-start
        max_render_time = max(max_render_time, end-start)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)
        # no data saving
        torchvision.utils.save_image(render_pkg["render"], os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        # if idx >= 50:
        #     break

    print(f"Number of Samples: {height_arr.shape[0]}")
    print(f'Average FPS: {height_arr.shape[0] /avg_render_time:.4f}')
    print(f'Min FPS: {1 / max_render_time:.4f}')
    print(f'Average Memory: {avg_memory / height_arr.shape[0]:.4f} M')
    print(f'Max Memory: {max_memory:.4f} M')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, load_vq : bool, skip_train : bool, skip_test : bool, custom_test : bool, pitch : float, heights : list):

    with torch.no_grad():
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs'])

        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)
        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if custom_test:
            views = scene.getTrainCameras() + scene.getTestCameras()
            render_set(dataset.model_path, filename, scene.loaded_iter, views, gaussians, pipeline, background, pitch, heights)
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, pitch, heights)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, pitch, heights)

def parse_cfg(cfg):
    lp = GroupParams()
    op = GroupParams()
    pp = GroupParams()

    for arg in cfg['model_params'].items():
        setattr(lp, arg[0], arg[1])
    
    for arg in cfg['optim_params'].items():
        setattr(op, arg[0], arg[1]) 

    for arg in cfg['pipeline_params'].items():
        setattr(pp, arg[0], arg[1])
    
    return lp, op, pp


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', type=str, help='train config file path of fused model')
    parser.add_argument('--model_path', type=str, help='model path of fused model')
    parser.add_argument("--custom_test", type=str, help="appointed test path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument("--pitch", type=float, default=-180.0)
    parser.add_argument("--min_height", type=float, default=5)
    parser.add_argument("--max_height", type=float, default=30)
    parser.add_argument("--num_samples", type=float, default=50)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.model_path is None:
        args.model_path = os.path.join('output', os.path.basename(args.config).split('.')[0])
    if args.load_vq:
        args.iteration = None
    args.heights = [args.min_height, args.max_height, args.num_samples]

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        setattr(lp, 'config_path', args.config)
        if lp.model_path == '':
            lp.model_path = args.model_path

    render_sets(lp, args.iteration, pp, args.load_vq, args.skip_train, args.skip_test, args.custom_test, args.pitch, args.heights)