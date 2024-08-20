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
import json
import torch
import torchvision
import time
import numpy as np
from tqdm import tqdm
from arguments import GroupParams
from scene import LargeScene
from scene.datasets import GSDataset
from os import makedirs
from gaussian_renderer import render_large
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.data import DataLoader
from utils.general_utils import parse_cfg

def render_set(model_path, name, iteration, gs_dataset, gaussians, pipeline, background):
    avg_render_time = 0
    max_render_time = 0
    avg_memory = 0
    max_memory = 0

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    data_loader = DataLoader(gs_dataset, batch_size=1, shuffle=False, num_workers=0)
    for idx, (cam_info, gt_image) in enumerate(tqdm(data_loader, desc="Rendering progress")):   
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        rendering = render_large(cam_info, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize()
        end = time.time()
        
        gt = gt_image[0:3, :, :]
        avg_render_time += end-start
        max_render_time = max(max_render_time, end-start)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    with open(model_path + "/costs.json", 'w') as fp:
        json.dump({
            "Average FPS": len(data_loader)/avg_render_time,
            "Min FPS": 1/max_render_time,
            "Average Memory(M)": avg_memory/len(data_loader),
            "Max Memory(M)": max_memory,
            "Number of Gaussians": gaussians.get_xyz.shape[0]
        }, fp, indent=True)
    
    print(f'Average FPS: {len(data_loader)/avg_render_time:.4f}')
    print(f'Min FPS: {1/max_render_time:.4f}')
    print(f'Average Memory: {avg_memory/len(data_loader):.4f} M')
    print(f'Max Memory: {max_memory:.4f} M')
    print(f'Number of Gaussians: {gaussians.get_xyz.shape[0]}')


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, load_vq : bool, skip_train : bool, skip_test : bool, custom_test : bool):

    with torch.no_grad():
        modules = __import__('scene')
        model_config = dataset.model_config
        gaussians = getattr(modules, model_config['name'])(dataset.sh_degree, **model_config['kwargs'])

        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)
        scene = LargeScene(dataset, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)
        print(f"Number of Gaussians: {gaussians.get_xyz.shape[0]}")

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if custom_test:
            views = scene.getTrainCameras() + scene.getTestCameras()
            gs_dataset = GSDataset(views, scene, dataset, pipeline)
            render_set(dataset.model_path, filename, scene.loaded_iter, gs_dataset, gaussians, pipeline, background)
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                gs_dataset = GSDataset(scene.getTrainCameras(), scene, dataset, pipeline)
                render_set(dataset.model_path, "train", scene.loaded_iter, gs_dataset, gaussians, pipeline, background)

            if not skip_test:
                gs_dataset = GSDataset(scene.getTestCameras(), scene, dataset, pipeline)
                render_set(dataset.model_path, "test", scene.loaded_iter, gs_dataset, gaussians, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', type=str, help='train config file path of fused model')
    parser.add_argument('--model_path', type=str, help='model path of fused model')
    parser.add_argument("--custom_test", type=str, help="appointed test path")
    parser.add_argument("--load_vq", action="store_true")
    parser.add_argument('--block_id', type=int, default=-1)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.model_path is None:
        args.model_path = os.path.join('output', os.path.basename(args.config).split('.')[0])
    if args.load_vq:
        args.iteration = 30000  # apply a default value

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg, args)

    render_sets(lp, args.iteration, pp, args.load_vq, args.skip_train, args.skip_test, args.custom_test)