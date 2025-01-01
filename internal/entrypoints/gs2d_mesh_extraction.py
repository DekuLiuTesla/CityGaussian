from dataclasses import dataclass
import os
import torch
import open3d as o3d
from jsonargparse import CLI
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gs2d_mesh_utils import GS2DMeshUtils, post_process_mesh


@dataclass
class CLIArgs:
    model_path: str

    dataset_path: str = None

    voxel_size: float = -1.

    depth_trunc: float = -1.

    sdf_trunc: float = -1.

    num_cluster: int = 50

    unbounded: bool = False

    mesh_res: int = 1024


def main():
    args = CLI(CLIArgs)

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
        if args.dataset_path is not None:
            dataset_path = args.dataset_path
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

    # set the active_sh to 0 to export only diffuse texture
    model.active_sh_degree = 0
    bg_color = torch.zeros((3,), dtype=torch.float, device=device)
    maps = GS2DMeshUtils.render_views(model, renderer, cameras, bg_color)
    bound = GS2DMeshUtils.estimate_bounding_sphere(cameras)

    if args.unbounded:
        name = 'fuse_unbounded.ply'
        mesh = GS2DMeshUtils.extract_mesh_unbounded(
            maps=maps,
            bound=bound,
            cameras=cameras,
            model=model,
            resolution=args.mesh_res,
        )
    else:
        name = 'fuse.ply'
        _, radius = bound
        depth_trunc = (radius * 2.0) if args.depth_trunc < 0 else args.depth_trunc
        voxel_size = (depth_trunc / args.mesh_res) if args.voxel_size < 0 else args.voxel_size
        sdf_trunc = 5.0 * voxel_size if args.sdf_trunc < 0 else args.sdf_trunc
        mesh = GS2DMeshUtils.extract_mesh_bounded(maps=maps, cameras=cameras, voxel_size=voxel_size, sdf_trunc=sdf_trunc, depth_trunc=depth_trunc)

    output_dir = args.model_path
    if os.path.isfile(output_dir):
        output_dir = os.path.dirname(output_dir)
    o3d.io.write_triangle_mesh(os.path.join(output_dir, name), mesh)
    print("mesh saved at {}".format(os.path.join(output_dir, name)))

    # post-process the mesh and save, saving the largest N clusters
    print("post-processing...")
    mesh_post = post_process_mesh(mesh, cluster_to_keep=args.num_cluster)
    o3d.io.write_triangle_mesh(os.path.join(output_dir, name.replace('.ply', '_post.ply')), mesh_post)
    print("mesh post processed saved at {}".format(os.path.join(output_dir, name.replace('.ply', '_post.ply'))))
