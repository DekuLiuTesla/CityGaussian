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
from gaussian_renderer import render_lod
from utils.general_utils import safe_state
from utils.large_utils import which_block, block_filtering
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from torch.utils.data import DataLoader
from utils.camera_utils import loadCamTA3

class BlockedGaussian:

    gaussians : GaussianModel

    def __init__(self, gaussians, lp, range=[0, 1], scale=1.0, compute_cov3D_python=False):
        self.cell_corners = []
        self.xyz = None
        self.feats = None
        self.max_sh_degree = lp.sh_degree
        self.device = gaussians.get_xyz.device
        self.compute_cov3D_python = compute_cov3D_python
        self.cell_ids = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.long, device=self.device)
        self.mask = torch.zeros(gaussians.get_opacity.shape[0], dtype=torch.bool, device=self.device)

        self.block_dim = lp.block_dim
        self.num_cell = lp.block_dim[0] * lp.block_dim[1] * lp.block_dim[2]
        self.aabb = lp.aabb
        self.scale = scale
        self.range = range

        self.cell_divider(gaussians)
        self.cell_corners = torch.stack(self.cell_corners, dim=0)

    def cell_divider(self, gaussians, n=4):
        with torch.no_grad():
            if self.compute_cov3D_python:
                geometry = gaussians.get_covariance(self.scale).to(self.device)
            else:
                geometry = torch.cat([gaussians.get_scaling,
                                      gaussians.get_rotation], dim=1)
            self.xyz = gaussians.get_xyz
            self.feats = torch.cat([gaussians.get_opacity,  
                                    gaussians.get_features.reshape(geometry.shape[0], -1),
                                    geometry], dim=1).half()
            
            for cell_idx in range(self.num_cell):
                cell_mask = block_filtering(cell_idx, self.xyz, self.aabb, self.block_dim, self.scale)
                self.cell_ids[cell_mask] = cell_idx
                # MAD to eliminate influence of outsiders
                xyz_median = torch.median(self.xyz[cell_mask], dim=0)[0]
                delta_median = torch.median(torch.abs(self.xyz[cell_mask] - xyz_median), dim=0)[0]
                xyz_min = xyz_median - n * delta_median
                xyz_min = torch.max(xyz_min, torch.min(self.xyz[cell_mask], dim=0)[0])
                xyz_max = xyz_median + n * delta_median
                xyz_max = torch.min(xyz_max, torch.max(self.xyz[cell_mask], dim=0)[0])
                corners = torch.tensor([[xyz_min[0], xyz_min[1], xyz_min[2]],
                                       [xyz_min[0], xyz_min[1], xyz_max[2]],
                                       [xyz_min[0], xyz_max[1], xyz_min[2]],
                                       [xyz_min[0], xyz_max[1], xyz_max[2]],
                                       [xyz_max[0], xyz_min[1], xyz_min[2]],
                                       [xyz_max[0], xyz_min[1], xyz_max[2]],
                                       [xyz_max[0], xyz_max[1], xyz_min[2]],
                                       [xyz_max[0], xyz_max[1], xyz_max[2]]], device=self.xyz.device)
                self.cell_corners.append(corners)
    
    def get_feats(self, indices, distances):
        out_xyz = torch.tensor([], device=self.device, dtype=self.xyz.dtype)
        out_feats = torch.tensor([], device=self.device, dtype=self.feats.dtype)
        block_mask = (distances >= self.range[0]) & (distances < self.range[1])
        if block_mask.sum() > 0:
            self.mask = torch.isin(self.cell_ids, indices[block_mask].to(self.device))
            out_xyz = self.xyz[self.mask]
            out_feats = self.feats[self.mask]
        return out_xyz, out_feats

    def get_feats_ptwise(self, viewpoint_cam):
        out_xyz = torch.tensor([], device=self.device, dtype=self.xyz.dtype)
        out_feats = torch.tensor([], device=self.device, dtype=self.feats.dtype)

        homo_xyz = torch.cat([self.xyz, torch.ones_like(self.xyz[..., [0]])], dim=-1)
        cam_center = viewpoint_cam.camera_center
        viewmatrix = viewpoint_cam.world_view_transform
        xyz_cam = homo_xyz @ viewmatrix
        self.mask = (xyz_cam[..., 2] > 0.2)
        if self.mask.sum() == 0:
            return out_xyz, out_feats

        distances = torch.norm(self.xyz - cam_center[None, :3], dim=-1)
        self.mask &= (distances >= self.range[0]) & (distances < self.range[1])
        if self.mask.sum() > 0:
            out_xyz = self.xyz[self.mask]
            out_feats = self.feats[self.mask]
        return out_xyz, out_feats

def load_gaussians(cfg, config_name, iteration=30_000, load_vq=False, device='cuda', source_path='data/matrix_city/aerial/test/block_all_test'):
    
    lp, op, pp = parse_cfg(cfg)
    setattr(lp, 'config_path', cfg)
    lp.source_path = source_path
    lp.model_path = os.path.join("output/", config_name)

    modules = __import__('scene')
    
    with torch.no_grad():
        if 'apply_voxelize' in lp.model_config['kwargs'].keys():
            lp.model_config['kwargs']['apply_voxelize'] = False
        gaussians = getattr(modules, lp.model_config['name'])(lp.sh_degree, device=device, **lp.model_config['kwargs'])
        scene = LargeScene(lp, gaussians, load_iteration=iteration, load_vq=load_vq, shuffle=False)
        print(f'Init {config_name} with {len(gaussians.get_opacity)} points\n')

    return gaussians, scene

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, pitch, traj_info):
    avg_render_time = 0
    max_render_time = 0
    avg_memory = 0
    max_memory = 0

    render_path = os.path.join(model_path, name, "ours_lod_TA3", "renders")
    gts_path = os.path.join(model_path, name, "ours_lod_TA3", "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    view = views[len(views) // 2]
    min_h, max_h, num_steps, radius = traj_info
    step = 1.0 / num_steps

    for t in tqdm(np.arange(0, 1, step), desc="Rendering progress"):
        xyz = np.zeros(3)
        xyz[0] = radius * np.cos(2 * np.pi * t)
        xyz[1] = radius * np.sin(2 * np.pi * t)
        xyz[2] = min_h + t * (max_h - min_h)

        delta = -xyz
        yaw = np.arctan2(-delta[0], delta[1])
        pitch = np.arctan2(-np.linalg.norm(delta[:2]), delta[2])

        viewpoint_cam = loadCamTA3(lp, int(t/step), view, 1.0, xyz=xyz, pitch=pitch, yaw=yaw)

        # gpu_tracker.track() 
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        start = time.time()
        rendering = render_lod(viewpoint_cam, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize()
        end = time.time()
        avg_render_time += end-start
        max_render_time = max(max_render_time, end-start)

        forward_max_memory_allocated = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        avg_memory += forward_max_memory_allocated
        max_memory = max(max_memory, forward_max_memory_allocated)
        # no data saving
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(int(t/step)) + ".png"))
        # break
        # torchvision.utils.save_image(viewpoint_cam.original_image[0:3, :, :], os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        # if idx >= 50:
        #     break

    print(f"Number of Samples: {num_steps}")
    print(f'Average FPS: {num_steps /avg_render_time:.4f}')
    print(f'Min FPS: {1 / max_render_time:.4f}')
    print(f'Average Memory: {avg_memory / num_steps:.4f} M')
    print(f'Max Memory: {max_memory:.4f} M')

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, load_vq : bool, skip_train : bool, skip_test : bool, custom_test : bool, pitch : float, traj_info : list):

    with torch.no_grad():
        config_2 = 'config/block_mc_aerial_block_all_lr_c36_loss_5_75_lr64_vq.yaml'
        config_name_2 = os.path.splitext(os.path.basename(config_2))[0]
        with open(config_2) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        lod_gs_2, scene = load_gaussians(cfg, config_name_2, iteration=None, load_vq=True)

        config_1 = 'config/block_mc_aerial_block_all_lr_c36_loss_5_66_lr64_vq.yaml'
        config_name_1 = os.path.splitext(os.path.basename(config_1))[0]
        with open(config_1) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        lod_gs_1, _ = load_gaussians(cfg, config_name_1, iteration=None, load_vq=True)

        config_0 = 'config/block_mc_aerial_block_all_lr_c36_loss_5_50_lr64_vq.yaml'
        config_name_0 = os.path.splitext(os.path.basename(config_0))[0]
        with open(config_0) as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        lod_gs_0, _ = load_gaussians(cfg, config_name_0, iteration=None, load_vq=True)

        with torch.no_grad():
            torch.cuda.empty_cache()
            lod_gs_0 = BlockedGaussian(lod_gs_0, lp, range=[0, 2.5], compute_cov3D_python=pp.compute_cov3D_python)
            lod_gs_1 = BlockedGaussian(lod_gs_1, lp, range=[2.5, 5], compute_cov3D_python=pp.compute_cov3D_python)
            lod_gs_2 = BlockedGaussian(lod_gs_2, lp, range=[5, 100], compute_cov3D_python=pp.compute_cov3D_python)
            torch.cuda.empty_cache()
        
        if custom_test:
            dataset.source_path = custom_test
            filename = os.path.basename(dataset.source_path)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if custom_test:
            views = scene.getTrainCameras() + scene.getTestCameras()
            render_set(dataset.model_path, filename, scene.loaded_iter, views, [lod_gs_0, lod_gs_1, lod_gs_2], pipeline, background, pitch, traj_info)
            print("Skip both train and test, render all views")
        else:
            if not skip_train:
                render_set(dataset.model_path, "train", scene.loaded_iter, views, [lod_gs_0, lod_gs_1, lod_gs_2], pipeline, background, pitch, traj_info)

            if not skip_test:
                render_set(dataset.model_path, "test", scene.loaded_iter, views, [lod_gs_0, lod_gs_1, lod_gs_2], pipeline, background, pitch, height)

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
    parser.add_argument("--radius", type=float, default=5)
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
    args.traj_info = [args.min_height, args.max_height, args.num_samples, args.radius]

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        setattr(lp, 'config_path', args.config)
        if lp.model_path == '':
            lp.model_path = args.model_path

    render_sets(lp, args.iteration, pp, args.load_vq, args.skip_train, args.skip_test, args.custom_test, args.pitch, args.traj_info)