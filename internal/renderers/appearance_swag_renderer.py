from typing import Any, Tuple, Optional
import torch
from internal.configs.appearance import AppearanceModelParams, SwagAppearanceModelParams
from internal.cameras.cameras import Camera
from internal.models.gaussian_model import GaussianModel
from internal.models.appearance_model import AppearanceModel, SwagAppearanceModel
from internal.utils.sh_utils import eval_sh
from .vanilla_renderer import VanillaRenderer


class AppearanceSWAGRenderer(VanillaRenderer):
    apply_on_gaussian: bool = False

    def __init__(
            self,
            appearance: SwagAppearanceModelParams,
            compute_cov3D_python: bool = False,
            convert_SHs_python: bool = False,
    ):
        super().__init__(compute_cov3D_python, convert_SHs_python)

        self.appearance_config = appearance

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            appearance: Tuple = None,
    ):
        # appearance
        if appearance is not None:
            override_color = appearance
        else:
            xyz = pc.get_xyz
            appearance_id = viewpoint_camera.appearance_id
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            with torch.no_grad():
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            override_color = torch.clamp_min(sh2rgb + 0.5, 0.0)
            override_color += self.appearance_model.get_appearance(xyz, override_color, appearance_id)
            override_color = torch.clamp(override_color, 0.0, 1.0)
        outputs = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, override_color)

        return outputs

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        appearance = self.appearance_config
        self.appearance_model = SwagAppearanceModel(
            n_input_dims=appearance.n_input_dims,
            n_appearance_count=appearance.n_appearance_count,
            n_appearance_dims=appearance.n_appearance_dims,
            n_neurons=appearance.n_neurons,
            n_hidden_layers=appearance.n_hidden_layers,
            color_activation=appearance.color_activation,
        )

    def training_setup(self, module) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        appearance_optimizer = torch.optim.Adam(
            [
                {"params": list(self.appearance_model.parameters()), "name": "appearance"}
            ],
            lr=self.appearance_config.optimization.lr,
            eps=self.appearance_config.optimization.eps,
        )
        appearance_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=appearance_optimizer,
            lr_lambda=lambda iter: self.appearance_config.optimization.gamma ** min(iter / self.appearance_config.optimization.max_steps, 1),
            verbose=False,
        )

        return appearance_optimizer, appearance_scheduler
