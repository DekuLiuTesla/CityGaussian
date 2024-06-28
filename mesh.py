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
import sys
import os
import yaml
from tqdm import tqdm
from os import makedirs
from argparse import ArgumentParser
from internal.utils.common import parse_cfg_yaml
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_block_dataparser import ColmapBlockDataParser
from internal.utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--model_path', type=str, help='path of 2DGS model')
    parser.add_argument('--config_path', type=str, default=None, help='path of configs')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)

    gaussians, renderer = GaussianModelLoader.search_and_load(
        args.model_path,
        sh_degree=3,
        device="cuda",
    )

    if args.config_path is None:
        config_path = os.path.join(args.model_path, "config.yaml")
    else:
        config_path = args.config_path
    load_from = GaussianModelLoader.search_load_file(args.model_path)
    with open(config_path, 'r') as f:
        config = parse_cfg_yaml(yaml.load(f, Loader=yaml.FullLoader))
    
    dataparser_outputs = ColmapBlockDataParser(
        os.path.expanduser(config.data.path),
        os.path.abspath(""),
        global_rank=0,
        params=config.data.params.colmap_block,
    ).get_outputs()
    
    mesh_dir = os.path.join(args.model_path, 'mesh', load_from.split('/')[-1].split('.')[0])
    gaussExtractor = GaussianExtractor(gaussians, renderer, bg_color=config.model.background_color)

    print("export mesh ...")
    os.makedirs(mesh_dir, exist_ok=True)
    # set the active_sh to 0 to export only diffuse texture
    gaussExtractor.gaussians.active_sh_degree = 0
    gaussExtractor.reconstruction(dataparser_outputs.train_set.cameras)
    # extract the mesh and save
    if args.unbounded:
        name = 'fuse_unbounded.ply'
        mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
    else:
        name = 'fuse.ply'
        depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
        voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
        sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
        mesh = gaussExtractor.extract_mesh_bounded(voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
    
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name), mesh)
    print("mesh saved at {}".format(os.path.join(mesh_dir, name)))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name.replace('.ply', '_post.ply')), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(mesh_dir, name.replace('.ply', '_post.ply'))))