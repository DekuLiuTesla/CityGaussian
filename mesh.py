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
from internal.utils.mesh_utils import GaussianExtractor, to_cam_open3d, post_process_mesh

import open3d as o3d

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument('--model_path', type=str, help='path of 2DGS model')
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument('--skip_mesh', action="store_true", help='Mesh: if directly apply post processing')
    parser.add_argument("--voxel_size", default=-1.0, type=float, help='Mesh: voxel size for TSDF')
    parser.add_argument("--depth_trunc", default=-1.0, type=float, help='Mesh: Max depth range for TSDF')
    parser.add_argument("--sdf_trunc", default=-1.0, type=float, help='Mesh: truncation value for TSDF')
    parser.add_argument("--sh_degree", default=3, type=int, help='Mesh: SH degree')
    parser.add_argument("--num_cluster", default=50, type=int, help='Mesh: number of connected clusters to export')
    parser.add_argument("--unbounded", action="store_true", help='Mesh: using unbounded mode for meshing')
    parser.add_argument("--tetrahedra", action="store_true", help='Mesh: whether use tetrahedra marching for unbounded scene')
    parser.add_argument("--downsample_factor", default=1, type=int, help='Mesh: downsample factor for tetrahedra marching')
    parser.add_argument("--use_trim_renderer", action="store_true", help='Mesh: whether to use trim renderer, suitable for original 3DGS')
    parser.add_argument('--mesh_name', type=str, default="fuse", help='Mesh: name of output mesh')
    parser.add_argument("--mesh_res", default=1024, type=int, help='Mesh: resolution for unbounded mesh extraction')
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)

    device = torch.device("cuda")

    # load ckpt
    loadable_file = GaussianModelLoader.search_load_file(args.model_path)
    print(loadable_file)
    dataparser_config = None
    if loadable_file.endswith(".ckpt"):
        ckpt = torch.load(loadable_file, map_location="cpu")
        # initialize model
        model = GaussianModelLoader.initialize_model_from_checkpoint(
            ckpt,
            device=device,
        )
        model.freeze()
        model.pre_activate_all_properties()
        # initialize renderer
        renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
            ckpt,
            stage="validate",
            device=device,
        )
        try:
            dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
        except:
            pass

        dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
    else:
        dataset_path = args.dataset_path
        if dataset_path is None:
            cfg_args_file = os.path.join(args.model_path, "cfg_args")
            try:
                from argparse import Namespace
                with open(cfg_args_file, "r") as f:
                    cfg_args = eval(f.read())
                dataset_path = cfg_args.source_path
            except Exception as e:
                print("Can not parse `cfg_args`: {}".format(e))
                print("Please specific the data path via: `--dataset_path`")
                exit(1)

        model, renderer = GaussianModelLoader.initialize_model_and_renderer_from_ply_file(
            loadable_file,
            device=device,
            eval_mode=True,
            pre_activate=True,
        )
    if dataparser_config is None:
        from internal.dataparsers.colmap_dataparser import Colmap
        dataparser_config = Colmap()

    # load dataset
    dataparser_outputs = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs()
    cameras = [i.to_device(device) for i in dataparser_outputs.train_set.cameras]
    
    if args.model_path.endswith('.ckpt'):
        mesh_dir = os.path.join(args.model_path.split('/checkpoints')[0], 'mesh')
    else:
        mesh_dir = os.path.join(args.model_path, 'mesh')
    gaussExtractor = GaussianExtractor(model, renderer, bg_color=ckpt["hyper_parameters"]["background_color"])

    if not args.skip_mesh:
        print(f"export mesh to {mesh_dir} ...")
        os.makedirs(mesh_dir, exist_ok=True)
        # set the active_sh to 0 to export only diffuse texture
        gaussExtractor.gaussians.active_sh_degree = 0
        
        # extract the mesh and save
        if args.unbounded:
            name = args.mesh_name + '_unbounded.ply'
            gaussExtractor.reconstruction(dataparser_outputs.train_set.cameras)
            if args.tetrahedra:
                mesh = gaussExtractor.extract_tetrahedra_mesh_unbounded(mesh_dir, ds_factor=args.downsample_factor, resolution=args.mesh_res)
            else:
                mesh = gaussExtractor.extract_mesh_unbounded(resolution=args.mesh_res)
        else:
            name = args.mesh_name + '.ply'
            depth_trunc = (gaussExtractor.radius * 2.0) if args.depth_trunc < 0  else args.depth_trunc
            voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
            sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
            # merge reconstruction and mesh extraction to save memory
            mesh = gaussExtractor.recon_extract_mesh_bounded(dataparser_outputs.train_set.cameras, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)
        
        o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name), mesh)
        print("mesh saved at {}".format(os.path.join(mesh_dir, name)))
    else:
        name = args.mesh_name + '.ply' if not args.unbounded else args.mesh_name + '_unbounded.ply'
        mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, args.mesh_name + '.ply'))
    # post-process the mesh and save, saving the largest N clusters
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(mesh_dir, name.replace('.ply', '_post.ply')), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(mesh_dir, name.replace('.ply', '_post.ply'))))