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
from scene import Scene
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, GroupParams, get_combined_args
from gaussian_renderer import GaussianModel

def render_set(model_path, name, subset_idx, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background)["render"]
        gt = view.original_image[0:3, :, :]
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:03d}_{1:05d}'.format(subset_idx, idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:03d}_{1:05d}'.format(subset_idx, idx) + ".png"))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, source_test : bool):
    # key parameters of dataset: sh_degree, model_path, iteration, sub_paths, 
    with torch.no_grad():
        gaussians_fuse = GaussianModel(dataset.sh_degree)
        gaussians_fuse.load_ply(os.path.join(dataset.model_path,
                                            "point_cloud",
                                            "iteration_" + str(iteration),
                                            "point_cloud.ply"))
        
        if not source_test:
            sub_paths = dataset.sub_paths
            for i, sub_path in enumerate(sub_paths):
                with open(sub_path) as f:
                    cfg = yaml.load(f, Loader=yaml.FullLoader)
                    lp, op, pp = parse_cfg(cfg)
                    if lp.model_path == '':
                        lp.model_path = os.path.join('output', os.path.basename(sub_path).split('.')[0])
                print("\nRendering with data from " + lp.model_path)

                gaussians = GaussianModel(dataset.sh_degree)
                scene = Scene(lp, gaussians, load_iteration=iteration, shuffle=False)

                bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

                if not skip_train:
                    render_set(dataset.model_path, "train", i, scene.loaded_iter, scene.getTrainCameras(), gaussians_fuse, pipeline, background)

                if not skip_test:
                    render_set(dataset.model_path, "test", i, scene.loaded_iter, scene.getTestCameras(), gaussians_fuse, pipeline, background)
        else:
            scene = Scene(dataset, gaussians_fuse, load_iteration=iteration, shuffle=False)
            bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            views = scene.getTrainCameras() + scene.getTestCameras()

            if skip_test or skip_train:
                print("Warning: skip_train and skip_test are ignored when source_test is set to True")
            
            render_set(dataset.model_path, "test_src", 0, scene.loaded_iter, views, gaussians_fuse, pipeline, background)


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
    parser.add_argument('--config_fuse', type=str, help='train config file path of fused model')
    parser.add_argument('--model_path', type=str, help='model path of fused model')
    parser.add_argument("--source_test", action="store_true", help="if test on data appointed by source_path")
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.model_path is None:
        args.model_path = os.path.join('output', os.path.basename(args.config_fuse).split('.')[0])

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    with open(args.config_fuse) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
        lp, op, pp = parse_cfg(cfg)
        setattr(lp, 'config_path', args.config_fuse)
        if lp.model_path == '':
            lp.model_path = args.model_path

    render_sets(lp, op.iterations, pp, args.skip_train, args.skip_test, args.source_test)