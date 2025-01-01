import add_pypath
import os
import torch
import argparse
from internal.utils.gaussian_model_loader import GaussianModelLoader


parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

ckpt_file = GaussianModelLoader.search_load_file(args.path)
ckpt = torch.load(ckpt_file, map_location="cpu")
model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")

# get fused properties
opacities, scales = model.get_3d_filtered_scales_and_opacities()
model.opacities = model.opacity_inverse_activation(opacities)
model.scales = model.scale_inverse_activation(scales)

# basic kwargs
model_init_kwargs = {
    "sh_degree": model.config.sh_degree,
}
renderer_init_kwargs = {
    "anti_aliased": True,
}

gaussian_model_name = ckpt["hyper_parameters"]["gaussian"].__class__.__name__
if gaussian_model_name == "AppearanceMipGaussian":
    print("AppearanceMipGaussian -> AppearanceModel & AppearanceRenderer")
    from internal.models.appearance_mip_gaussian import AppearanceFeatureGaussian
    from internal.renderers.gsplat_appearance_embedding_renderer import GSplatAppearanceEmbeddingRenderer
    model_class = AppearanceFeatureGaussian
    renderer_class = GSplatAppearanceEmbeddingRenderer

    model_init_kwargs.update({
        "appearance_feature_dims": model.config.appearance_feature_dims,
    })
    renderer_init_kwargs.update({
        "anti_aliased": True,
        "filter_2d_kernel_size": ckpt["hyper_parameters"]["renderer"].filter_2d_kernel_size,
        "model": ckpt["hyper_parameters"]["renderer"].model,
        "optimization": ckpt["hyper_parameters"]["renderer"].optimization,
        "tile_based_culling": getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
    })
elif gaussian_model_name == "MipSplatting":
    print("MipSplatting -> VanillaModel & GSplatRenderer")
    from internal.models.vanilla_gaussian import VanillaGaussian
    from internal.renderers.gsplat_v1_renderer import GSplatV1Renderer
    model_class = VanillaGaussian
    renderer_class = GSplatV1Renderer

    renderer_init_kwargs.update({
        "block_size": ckpt["hyper_parameters"]["renderer"].block_size,
        "anti_aliased": ckpt["hyper_parameters"]["renderer"].anti_aliased,
        "filter_2d_kernel_size": ckpt["hyper_parameters"]["renderer"].filter_2d_kernel_size,
        "separate_sh": getattr(ckpt["hyper_parameters"]["renderer"], "separate_sh", False),
        "tile_based_culling": getattr(ckpt["hyper_parameters"]["renderer"], "tile_based_culling", False),
    })
else:
    raise ValueError("unsupported model type '{}'".format(gaussian_model_name))

# update ckpt's model
ckpt["hyper_parameters"]["gaussian"] = model_class(**model_init_kwargs)

# update ckpt's renderer
ckpt["hyper_parameters"]["renderer"] = renderer_class(**renderer_init_kwargs)

# update model's state_dict
for k in list(ckpt["state_dict"].keys()):
    if k.startswith("gaussian_model.gaussians."):
        del ckpt["state_dict"][k]
model_state_dict = model.state_dict()
for k in ckpt["hyper_parameters"]["gaussian"].instantiate().get_property_names():
    ckpt["state_dict"]["gaussian_model.gaussians.{}".format(k)] = model_state_dict["gaussians.{}".format(k)]


ckpt["optimizer_states"] = []

output_path = os.path.join(os.path.dirname(ckpt_file), "mip_fused.ckpt")
torch.save(
    ckpt,
    output_path,
)
print(output_path)
