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
from tqdm import tqdm
from arguments import GroupParams
from scene import LargeScene
from os import makedirs
from gaussian_renderer import render_lod
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams
from scene.gaussian_model import GatheredGaussian, BlockedGaussian
from utils.large_utils import contract_to_unisphere, get_default_aabb
from utils.camera_utils import loadCam
from utils.general_utils import parse_cfg

def load_gaussians(lp, iteration=30_000, load_vq=False, device='cuda'):

    modules = __import__('scene')
    
    with torch.no_grad():
        gaussians = getattr(modules, lp.model_config['name'])(lp.sh_degree, device=device, **lp.model_config['kwargs'])
        scene = LargeScene(lp, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)
    return gaussians, scene

def render_set(lp, model_path, name, iteration, views, model, max_sh_degree, pipeline, background):
    avg_render_time = 0
    max_render_time = 0
    avg_memory = 0
    max_memory = 0

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    print(f"Save Path: {os.path.join(model_path, name)}")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx in tqdm(range(len(views)), desc="Rendering progress"):
        
        viewpoint_cam = loadCam(lp, idx, views[idx], 1.0)

        # gpu_tracker.track() 
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        rendering = render_lod(viewpoint_cam, model, pipeline, background)["render"]
        torch.cuda.synchronize()
        end = time.time()
        avg_render_time += end-start
        max_render_time = max(max_render_time, end-start)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)
        # data saving
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(viewpoint_cam.original_image[0:3, :, :], os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
    
    with open(model_path + "/costs.json", 'w') as fp:
        json.dump({
            "Average FPS": len(views)/avg_render_time,
            "Min FPS": 1/max_render_time,
            "Average Memory(M)": avg_memory/len(views),
            "Max Memory(M)": max_memory,
        }, fp, indent=True)
    
    print(f'Average FPS: {len(views)/avg_render_time:.4f}')
    print(f'Min FPS: {1/max_render_time:.4f}')
    print(f'Average Memory: {avg_memory/len(views):.4f} M')
    print(f'Max Memory: {max_memory:.4f} M')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, load_vq : bool, skip_train : bool, skip_test : bool, custom_test : bool):

    with torch.no_grad():
        if dataset.aabb is None:
            import numpy as np
            dataset.aabb = np.load(os.path.join(dataset.source_path, "data_partitions", f"{dataset.partition_name}_aabb.npy")).tolist()
            print(f"Use default AABB of {[round(x, 2) for x in dataset.aabb]}")

        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)

        lod_gs_list = []
        org_model_path = dataset.model_path
        for i in range(len(dataset.lod_configs)):
            dataset.model_path = dataset.lod_configs[i]
            lod_gs, scene = load_gaussians(dataset, iteration, load_vq)
            print(f"Init LoD {len(dataset.lod_configs)-i} with {lod_gs.get_xyz.shape[0]} points from {dataset.model_path}")
            lod_gs = BlockedGaussian(lod_gs, dataset, compute_cov3D_python=pp.compute_cov3D_python)
            lod_gs_list.append(lod_gs)
        dataset.model_path = org_model_path
        
        num_cell, max_sh_degree = lod_gs_list[-1].num_cell, lod_gs_list[-1].max_sh_degree
        loaded_iter, train_cams, test_cams = scene.loaded_iter, scene.getTrainCameras(), scene.getTestCameras()
        del lod_gs, scene

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if custom_test:
            views = train_cams + test_cams
            render_set(dataset, dataset.model_path, filename, loaded_iter, views, lod_gs_list, max_sh_degree, pipeline, background)
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                render_set(dataset, dataset.model_path, "train", loaded_iter, train_cams, lod_gs_list, max_sh_degree, pipeline, background)

            if not skip_test:
                render_set(dataset, dataset.model_path, "test", loaded_iter, test_cams, lod_gs_list, max_sh_degree, pipeline, background)


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--config', type=str, help='train config file path of fused model')
    parser.add_argument('--model_path', type=str, help='model path of fused model')
    parser.add_argument("--custom_test", type=str, help="appointed test path")
    parser.add_argument("--load_vq", action="store_true")
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
        print(f'Render with resolution {lp.resolution}\n')

    render_sets(lp, args.iteration, pp, args.load_vq, args.skip_train, args.skip_test, args.custom_test)
