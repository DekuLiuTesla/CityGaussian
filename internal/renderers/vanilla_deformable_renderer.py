import os.path
import torch
from .renderer import Renderer
from .vanilla_renderer import VanillaRenderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from ..models.vanilla_deform_model import VanillaDeformNetwork
from ..utils.rigid_utils import from_homogenous, to_homogenous
from ..utils.common import parse_cfg_args


class VanillaDeformableRenderer(Renderer):
    def __init__(
            self,
            model_path: str,
            load_iteration: int,
            device,
    ) -> None:
        super().__init__()

        cfg_args = self._parse_cfg_args(model_path)

        self.deform_model = VanillaDeformNetwork(
            is_blender=cfg_args.is_blender,
            is_6dof=cfg_args.is_6dof,
        )

        checkpoint = torch.load(os.path.join(model_path, "deform", "iteration_{}".format(load_iteration), "deform.pth"))
        self.deform_model.load_state_dict(checkpoint)

        self.deform_model = self.deform_model.to(device)

        self.is_6dof = cfg_args.is_6dof

    @classmethod
    def _parse_cfg_args(cls, model_path):
        return parse_cfg_args(os.path.join(model_path, "cfg_args"))

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        # without ast noise
        N = pc.get_xyz.shape[0]
        time_input = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
        d_xyz, d_rotation, d_scaling = self.deform_model(pc.get_xyz.detach(), time_input)

        return self._render(
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
        )

    def _render(
            self,
            d_xyz,
            d_rotation,
            d_scaling,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
    ):
        if self.is_6dof is True:
            if torch.is_tensor(d_xyz) is False:
                means3D = pc.get_xyz
            else:
                means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
        else:
            means3D = pc.get_xyz + d_xyz

        opacity = pc.get_opacity
        scales = pc.get_scaling + d_scaling
        rotations = pc.get_rotation + d_rotation
        features = pc.get_features

        return VanillaRenderer.render(
            means3D=means3D,
            opacity=opacity,
            scales=scales,
            rotations=rotations,
            features=features,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
        )
