import os
import sys
import yaml
import torch
import add_pypath
import json
from tqdm import tqdm
from argparse import ArgumentParser, Namespace

from internal.utils.general_utils import parse
from internal.utils.sh_utils import SH2RGB
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.citygs_partitioning_utils import CityGSSceneConfig, CityGSPartitionableScene
from internal.dataparsers.colmap_dataparser import ColmapDataParser

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument("--config_path", type=str, help="path of finetuned model", default=None)
    parser.add_argument("--model_path", type=str, help="path of coarse global model")
    parser.add_argument("--contract", action="store_true", help="whether partition in contracted space, suitable for irregular distribution")
    parser.add_argument("--reorient", action="store_true", help="whether reorient the scene before partitioning")
    parser.add_argument("--content_threshold", type=float, default=0.08)
    parser.add_argument("--block_dim", type=int, nargs="+", default=None)
    parser.add_argument("--aabb", type=float, nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        '--origin',
        type=lambda v: "auto" if v.lower() == "auto" else [float(x) for x in v.split(',')],
        help="origin tensor. enter numbers separated by comma or \"auto\""
    )
    args = parser.parse_args(sys.argv[1:])

    if args.config_path is not None:
        # parameters in config file will overwrite command line arguments
        print(f"Loading parameters according to config file {args.config_path}")
        with open(args.config_path, "r") as f:
            config = parse(yaml.load(f, Loader=yaml.FullLoader))
            params = config.data.parser.init_args
        args.block_dim = params.block_dim
        args.content_threshold = params.content_threshold if hasattr(params, "content_threshold") else args.content
        if args.contract:
            args.aabb = params.aabb if hasattr(params, "aabb") else args.aabb
        
        ckpt_path = config.model.initialize_from
        if "point_cloud" in config.model.initialize_from:
            args.model_path = config.model.initialize_from.split("/point_cloud/")[0]
        elif "checkpoints" in config.model.initialize_from:
            args.model_path = config.model.initialize_from.split("/checkpoints/")[0]
    
    # initialize model
    device = torch.device("cuda")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    bkgd_color = ckpt["hyper_parameters"]["background_color"]
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
    print("Gaussian count: {}".format(model.get_xyz.shape[0]))

    # initialize dataset
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    dataset_path = ckpt["datamodule_hyper_parameters"]["path"]
    dataset = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    ).get_outputs().train_set
    
    c2w = torch.linalg.inv(torch.stack([cam.world_to_camera.T for cam in dataset.cameras]))
    if args.reorient:
        # reorient the scene, calculate the up direction of the scene
        # NOTE: 
        #   the calculated direction may not be perfect or even incorrect sometimes, 
        #   in such a situation, you need to provide a correct up vector
        up = -torch.mean(c2w[:, :3, 1], dim=0)
        up = up / torch.linalg.norm(up)

        rotation = ColmapDataParser.rotation_matrix(up, torch.tensor([0, 0, 1], dtype=up.dtype))
        rotation_transform = torch.eye(4, dtype=up.dtype)
        rotation_transform[:3, :3] = rotation

        reoriented_camera_centers = c2w[:, :3, 3] @ rotation_transform[:3, :3].T
        reoriented_point_cloud_xyz = model.get_xyz.to(c2w.device) @ rotation_transform[:3, :3].T
    else:
        up = torch.tensor([0., 0., 1.], dtype=c2w.dtype)
        rotation_transform = torch.eye(4, dtype=c2w.dtype)
        reoriented_camera_centers = c2w[:, :3, 3]
        reoriented_point_cloud_xyz = model.get_xyz.to(c2w.device)
    point_rgbs = SH2RGB(model.get_features[:, 0]).clamp(0, 1).to(c2w.device) * 255.0

    scene_config = CityGSSceneConfig(
        origin=torch.tensor([0., 0.]),
        block_dim=args.block_dim,
        aabb=args.aabb,
        contract=args.contract,
        content_threshold=args.content_threshold,
    )
    scene = CityGSPartitionableScene(scene_config, reoriented_camera_centers[..., :2], reoriented_points=reoriented_point_cloud_xyz[..., :2])
    
    scene.get_bounding_box_by_points()
    if args.origin:
        if args.origin == 'auto':
            scene_config.origin = (scene.point_based_bounding_box.min + scene.point_based_bounding_box.max) / 2
        else:
            scene_config.origin = torch.tensor(args.origin)
    scene.get_scene_bounding_box()
    scene.build_partition_coordinates()
    print(f"Camera center based partition assignment: {scene.camera_center_based_partition_assignment().sum(-1)}")
    print(f"Projection based partition assignment: {scene.projection_based_partition_assignment(model, renderer, dataset.cameras, bkgd_color).sum(-1)}")

    output_path = os.path.join(dataset_path, scene.build_output_dirname())
    if not args.force and os.path.exists(output_path):
        try:
            assert os.path.exists(os.path.join(output_path, "partitions.pt")) is False, "Partition data already exists, please use --force to overwrite"
            if not os.path.exists(output_path):
                os.makedirs(output_path)
        except:
            del output_path
            raise
    print(f"Output path: {output_path}")

    scene.save(
        output_path,
        extra_data={
            "up": up,
            "rotation_transform": rotation_transform,
        }
    )

    scene.save_plot(scene.plot_partitions, os.path.join(output_path, "partitions.png"), notebook=False)

    is_images_assigned_to_partitions = torch.logical_or(scene.is_camera_in_partition, scene.is_partitions_visible_to_cameras)
    print(f"Overall images assigned to partitions: {is_images_assigned_to_partitions.sum(-1)}")

    max_plot_points = 51_200
    plot_point_sparsify = max(reoriented_point_cloud_xyz.shape[0] // max_plot_points, 1)

    written_idx_list = []
    for partition_idx in tqdm(list(range(is_images_assigned_to_partitions.shape[0]))):
        partition_image_indices = is_images_assigned_to_partitions[partition_idx].nonzero().squeeze(-1).tolist()
        if len(partition_image_indices) == 0:
            continue
            
        written_idx_list.append(partition_idx)
            
        camera_list = []
        
        with open(os.path.join(output_path, "{}.txt".format(scene.partition_coordinates.get_str_id(partition_idx))), "w") as f:
            for image_index in partition_image_indices:
                f.write(dataset.image_names[image_index])
                f.write("\n")
                
                # below camera list is just for visualization, not for training, so its camera intrinsics are fixed values
                color = [0, 0, 255]
                if scene.is_partitions_visible_to_cameras[partition_idx][image_index]:
                    color = [255, 0, 0]
                camera_list.append({
                    "id": image_index,
                    "img_name": dataset.image_names[image_index],
                    "width": 1920,
                    "height": 1080,
                    "position": c2w[image_index][:3, 3].numpy().tolist(),
                    "rotation": c2w[image_index][:3, :3].numpy().tolist(),
                    "fx": 1600,
                    "fy": 1600,
                    "color": color,
                })
                
        with open(os.path.join(
                output_path, 
                f"cameras-{scene.partition_coordinates.get_str_id(partition_idx)}.json",
        ), "w") as f:
            json.dump(camera_list, f, indent=4, ensure_ascii=False)
        
        scene.save_plot(
            scene.plot_partition_assigned_cameras,
            os.path.join(output_path, "{}.png".format(scene.partition_coordinates.get_str_id(partition_idx))),
            False,
            partition_idx,
            reoriented_point_cloud_xyz,
            point_rgbs,
            point_sparsify=plot_point_sparsify,
        )

    max_store_points = 512_000
    store_point_step = max(point_rgbs.shape[0] // max_store_points, 1)
    from internal.utils.graphics_utils import store_ply
    store_ply(os.path.join(output_path, "points.ply"), model.get_xyz.to('cpu')[::store_point_step], point_rgbs[::store_point_step])

    print("Run below commands to visualize the partitions in web viewer:\n")
    for partition_idx in written_idx_list:
        id_str = scene.partition_coordinates.get_str_id(partition_idx)
        print("python utils/show_cameras.py \\\n    '{}' \\\n    --points='{}' \\\n --up {:.3f} {:.3f} {:.3f} \n".format(
            os.path.join(output_path, "cameras-{}.json".format(id_str)),
            os.path.join(output_path, "points.ply"),
            *(up.tolist()),
        ))
    
    # All done
    print("Partition complete.")