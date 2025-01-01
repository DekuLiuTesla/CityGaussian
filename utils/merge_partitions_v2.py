import add_pypath
import os
import sys
import gc
import json
import argparse
import torch
from tqdm.auto import tqdm
from trained_partition_utils import get_trained_partitions, split_partition_gaussians
from internal.cameras.cameras import Camera
from internal.dataparsers.colmap_dataparser import Colmap
from internal.models.vanilla_gaussian import VanillaGaussian
from internal.models.appearance_feature_gaussian import AppearanceFeatureGaussianModel
from internal.models.mip_splatting import MipSplattingModelMixin
from internal.renderers.vanilla_renderer import VanillaRenderer
from internal.renderers.gsplat_v1_renderer import GSplatV1Renderer
from internal.renderers.gsplat_mip_splatting_renderer_v2 import GSplatMipSplattingRendererV2
from internal.density_controllers.vanilla_density_controller import VanillaDensityController
from internal.utils.gaussian_model_loader import GaussianModelLoader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("partition_dir")
    parser.add_argument("--project", "-p", type=str, required=False,
                        help="Project Name")
    parser.add_argument("--output_path", "-o", type=str, required=False)
    parser.add_argument("--min-images", type=int, default=32)
    parser.add_argument("--preprocess", action="store_true")
    args = parser.parse_args()

    if args.output_path is None:
        args.output_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", args.project, "merged.ckpt")
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    elif not args.output_path.endswith(".ckpt"):
        args.output_path += ".ckpt"

    if not args.preprocess:
        assert os.path.exists(args.output_path) is False, "output file '{}' already exists".format(args.output_path)

    return args


def fuse_appearance_features(ckpt: dict, gaussian_model, cameras_json: list[dict], image_name_to_camera: dict[str, Camera]):
    cuda_device = torch.device("cuda")
    gaussian_device = gaussian_model.get_means().device
    gaussian_model.to(device=cuda_device)
    renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(ckpt, "validation", cuda_device)

    cameras = [image_name_to_camera[i["img_name"]] for i in cameras_json]

    from fuse_appearance_embeddings_into_shs_dc import prune_and_get_weights, average_embedding_fusing
    n_average_cameras = 32
    _, visibility_score_pruned_sorted_indices, visibility_score_pruned_top_k_pdf = prune_and_get_weights(
        gaussian_model=gaussian_model,
        cameras=cameras,
        n_average_cameras=n_average_cameras,
        weight_device=cuda_device,
    )
    rgb_offset = average_embedding_fusing(
        gaussian_model,
        renderer,
        n_average_cameras=n_average_cameras,
        cameras=cameras,
        visibility_score_pruned_sorted_indices=visibility_score_pruned_sorted_indices,
        visibility_score_pruned_top_k_pdf=visibility_score_pruned_top_k_pdf,
        view_dir_average_mode="view_direction",
    )

    from internal.utils.sh_utils import RGB2SH
    sh_offset = RGB2SH(rgb_offset)
    gaussian_model.shs_dc = gaussian_model.shs_dc + sh_offset.unsqueeze(1).to(device=gaussian_model.shs_dc.device)
    gaussian_model.to(device=gaussian_device)


def update_ckpt(ckpt, merged_gaussians, max_sh_degree):
    # replace `AppearanceFeatureGaussian` with `VanillaGaussian`
    ckpt["hyper_parameters"]["gaussian"] = VanillaGaussian(sh_degree=max_sh_degree)

    # remove `GSplatAppearanceEmbeddingRenderer`'s states from ckpt
    state_dict_key_to_delete = []
    for i in ckpt["state_dict"]:
        if i.startswith("renderer."):
            state_dict_key_to_delete.append(i)
    for i in state_dict_key_to_delete:
        del ckpt["state_dict"][i]

    # replace `GSplatAppearanceEmbeddingRenderer` with `GSPlatRenderer`
    anti_aliased = True
    kernel_size = 0.3
    if isinstance(ckpt["hyper_parameters"]["renderer"], VanillaRenderer):
        anti_aliased = False
    elif isinstance(ckpt["hyper_parameters"]["renderer"], GSplatMipSplattingRendererV2) or ckpt["hyper_parameters"]["renderer"].__class__.__name__ == "GSplatAppearanceEmbeddingMipRenderer":
        kernel_size = ckpt["hyper_parameters"]["renderer"].filter_2d_kernel_size
    ckpt["hyper_parameters"]["renderer"] = GSplatV1Renderer(
        anti_aliased=anti_aliased,
        filter_2d_kernel_size=kernel_size,
        separate_sh=getattr(ckpt["hyper_parameters"]["renderer"], "separate_sh", True),
        tile_based_culling=getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
    )

    # remove existing Gaussians from ckpt
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith("gaussian_model.gaussians.") or i.startswith("frozen_gaussians."):
            del ckpt["state_dict"][i]

    # remove optimizer states
    ckpt["optimizer_states"] = []

    # reinitialize density controller states
    if isinstance(ckpt["hyper_parameters"]["density"], VanillaDensityController):
        for k in list(ckpt["state_dict"].keys()):
            if k.startswith("density_controller."):
                ckpt["state_dict"][k] = torch.zeros((merged_gaussians["means"].shape[0], *ckpt["state_dict"][k].shape[1:]), dtype=ckpt["state_dict"][k].dtype)

    # add merged gaussians to ckpt
    for k, v in merged_gaussians.items():
        ckpt["state_dict"]["gaussian_model.gaussians.{}".format(k)] = v


def fuse_mip_filters(gaussian_model):
    new_opacities, new_scales = gaussian_model.get_3d_filtered_scales_and_opacities()
    gaussian_model.opacities = gaussian_model.opacity_inverse_activation(new_opacities)
    gaussian_model.scales = gaussian_model.scale_inverse_activation(new_scales)


def main():
    """
    Overall pipeline:
        * Load the partition data
        * Get trainable partitions and their checkpoint filenames
        * For each partition
          * Load the checkpoint
          * Extract Gaussians falling into the partition bounding box
          * Fuse appearance features into SHs
        * Merge all extracted Gaussians
        * Update the checkpoint
          * Replace GaussianModel with the vanilla one
          * Replace `AppearanceEmbeddingRenderer` with the `GSPlatRenderer`
          * Clear optimizers' states
          * Re-initialize density controller's states
          * Replace with merged Gaussians
        * Saving
    """

    MERGABLE_PROPERTY_NAMES = ["means", "shs_dc", "shs_rest", "scales", "rotations", "opacities"]

    args = parse_args()

    torch.autograd.set_grad_enabled(False)

    partition_training, mergable_partitions, orientation_transformation = get_trained_partitions(
        partition_dir=args.partition_dir,
        project_name=args.project,
        min_images=args.min_images,
    )

    image_name_to_camera = None

    gaussians_to_merge = {}

    with tqdm(mergable_partitions, desc="Pre-processing") as t:
        for partition_idx, partition_id_str, ckpt_file, bounding_box in t:
            t.set_description("{}".format(partition_id_str))
            t.set_postfix_str("Loading checkpoint...")
            ckpt = torch.load(ckpt_file, map_location="cpu")

            t.set_postfix_str("Splitting...")
            gaussian_model, _, _ = split_partition_gaussians(
                ckpt,
                bounding_box,
                orientation_transformation,
            )

            if isinstance(gaussian_model, AppearanceFeatureGaussianModel):
                with open(os.path.join(
                        os.path.dirname(os.path.dirname(ckpt_file)),
                        "cameras.json"
                ), "r") as f:
                    cameras_json = json.load(f)

                # the dataset will only be loaded once
                if image_name_to_camera is None:
                    t.set_postfix_str("Loading colmap model...")

                    dataparser_config = Colmap(
                        split_mode="reconstruction",
                        eval_step=64,
                        points_from="random",
                    )
                    for i in ["image_dir", "mask_dir", "scene_scale", "reorient", "appearance_groups", "down_sample_factor", "down_sample_rounding_mode"]:
                        setattr(dataparser_config, i, getattr(ckpt["datamodule_hyper_parameters"]["parser"], i))
                    dataparser_outputs = dataparser_config.instantiate(
                        path=partition_training.dataset_path,
                        output_path=os.getcwd(),
                        global_rank=0,
                    ).get_outputs()

                    image_name_to_camera = {}
                    for idx in range(len(dataparser_outputs.train_set)):
                        image_name = dataparser_outputs.train_set.image_names[idx]
                        camera = dataparser_outputs.train_set.cameras[idx]
                        image_name_to_camera[image_name] = camera

                t.set_postfix_str("Fusing...")
                fuse_appearance_features(
                    ckpt,
                    gaussian_model,
                    cameras_json,
                    image_name_to_camera=image_name_to_camera,
                )

            if isinstance(gaussian_model, MipSplattingModelMixin):
                t.set_postfix_str("Fusing MipSplatting filters...")
                fuse_mip_filters(gaussian_model)

            if args.preprocess:
                update_ckpt(ckpt, {k: gaussian_model.get_property(k) for k in MERGABLE_PROPERTY_NAMES}, gaussian_model.max_sh_degree)
                torch.save(ckpt, os.path.join(
                    os.path.dirname(os.path.dirname(ckpt_file)),
                    "preprocessed.ckpt",
                ))
            else:
                for i in MERGABLE_PROPERTY_NAMES:
                    gaussians_to_merge.setdefault(i, []).append(gaussian_model.get_property(i))

    if args.preprocess:
        return

    # merge
    print("Merging...")
    merged_gaussians = {}
    for k, v in gaussians_to_merge.items():
        merged_gaussians[k] = torch.concat(v, dim=0)
        # release merged one to avoid OOM
        v.clear()
        gc.collect()
        torch.cuda.empty_cache()

    update_ckpt(ckpt, merged_gaussians, gaussian_model.max_sh_degree)

    # save
    print("Saving...")
    torch.save(ckpt, args.output_path)
    print("Saved to '{}'".format(args.output_path))

    viewer_args = ["python", "viewer.py", args.output_path]
    if orientation_transformation is not None:
        viewer_args += ["--up"]
        viewer_args += ["{:.4f}".format(i) for i in (-1. * partition_training.scene["extra_data"]["up"]).tolist()]
    print("The command to start web viewer:"
          " {}".format(" ".join(viewer_args)))


def test_main():
    sys.argv = [
        __file__,
        os.path.expanduser("~/dataset/JNUCar_undistorted/colmap/drone/dense_max_2048/0/partitions-size_3.0-enlarge_0.1-visibility_0.9_0.1"),
        "-p", "JNUAerial-0820",
    ]
    main()


if __name__ == "__main__":
    main()
