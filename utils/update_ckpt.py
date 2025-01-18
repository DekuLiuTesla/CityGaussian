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
    parser.add_argument("--project", "-p", type=str, required=False,
                        help="Project Name")
    args = parser.parse_args()

    args.ckpt_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", args.project)

    return args


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

    ckpt_file = GaussianModelLoader.search_load_file(args.ckpt_dir)
    ckpt = torch.load(ckpt_file, map_location="cpu")
    gaussian_model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, device="cpu")
    update_ckpt(ckpt, {k: gaussian_model.get_property(k) for k in MERGABLE_PROPERTY_NAMES}, gaussian_model.max_sh_degree)
    torch.save(ckpt, os.path.join(
        os.path.dirname(ckpt_file),
        "preprocessed.ckpt",
    ))
    print("Saved to", os.path.join(
        os.path.dirname(ckpt_file),
        "preprocessed.ckpt",
    ))


def test_main():
    sys.argv = [
        __file__,
        os.path.expanduser("~/dataset/JNUCar_undistorted/colmap/drone/dense_max_2048/0/partitions-size_3.0-enlarge_0.1-visibility_0.9_0.1"),
        "-p", "JNUAerial-0820",
    ]
    main()


if __name__ == "__main__":
    main()
