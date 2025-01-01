import os.path

import torch
from ..cameras import Camera
from ..model_components.gs4d_deformation import deform_network
from ..models.gaussian import GaussianModel
from ..utils.common import parse_cfg_args
from .renderer import Renderer
from .vanilla_renderer import VanillaRenderer


class VanillaGS4DRenderer(Renderer):
    def __init__(
            self,
            model_path: str,
            load_iteration: int,
            device,
    ):
        super().__init__()

        self.model_path = model_path
        self.load_iteration = load_iteration
        self.device = device

        self.compute_cov3D_python = False
        self.convert_SHs_python = False

        self.cfg_args = parse_cfg_args(os.path.join(model_path, "cfg_args"))

        checkpoint_dir = os.path.join(
            os.path.join(self.model_path),
            "point_cloud",
            "iteration_{}".format(load_iteration),
        )
        self.deformation = deform_network(self.cfg_args).to(device)
        self.deformation.load_state_dict(torch.load(os.path.join(
            checkpoint_dir,
            "deformation.pth",
        ), map_location=device))
        self.deformation_table = torch.load(os.path.join(checkpoint_dir, "deformation_table.pth"), map_location=device)

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        means3D = pc.get_xyz
        scales = pc.scales
        rotations = pc.rotations
        opacity = pc.opacities
        shs = pc.get_features
        time = viewpoint_camera.time.repeat(means3D.shape[0], 1)

        # deformation_point = self.deformation_table
        means3D_final, scales_final, rotations_final, opacity_final, shs_final = self.deformation(
            means3D,
            scales,
            rotations,
            opacity,
            shs,
            time,
        )

        # means3D_final = torch.zeros_like(means3D)
        # rotations_final = torch.zeros_like(rotations)
        # scales_final = torch.zeros_like(scales)
        # opacity_final = torch.zeros_like(opacity)
        # means3D_final[deformation_point] = means3D_deform
        # rotations_final[deformation_point] = rotations_deform
        # scales_final[deformation_point] = scales_deform
        # opacity_final[deformation_point] = opacity_deform
        # means3D_final[~deformation_point] = means3D[~deformation_point]
        # rotations_final[~deformation_point] = rotations[~deformation_point]
        # scales_final[~deformation_point] = scales[~deformation_point]
        # opacity_final[~deformation_point] = opacity[~deformation_point]

        scales_final = pc.scale_activation(scales_final)
        rotations_final = pc.rotation_activation(rotations_final)
        opacity_final = pc.opacity_activation(opacity)

        return VanillaRenderer.render(
            means3D=means3D_final,
            opacity=opacity_final,
            scales=scales_final,
            rotations=rotations_final,
            features=shs_final,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
        )
