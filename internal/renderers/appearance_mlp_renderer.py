from typing import Any, Tuple, Optional
import torch
from internal.configs.appearance import AppearanceModelParams
from internal.cameras.cameras import Camera
from internal.models.gaussian import GaussianModel
from internal.models.appearance_model import AppearanceModel
from internal.utils.sh_utils import eval_sh
from .vanilla_renderer import VanillaRenderer


class AppearanceMLPRenderer(VanillaRenderer):
    apply_on_gaussian: bool = False

    def __init__(
            self,
            appearance: AppearanceModelParams,
            apply_on_gaussian: bool = False,
            compute_cov3D_python: bool = False,
            convert_SHs_python: bool = False,
    ):
        super().__init__(compute_cov3D_python, convert_SHs_python)

        self.appearance_config = appearance
        self.apply_on_gaussian = apply_on_gaussian

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            appearance: Tuple = None,
            **kwargs,
    ):
        # appearance
        if appearance is not None:
            grayscale_factors, gamma = appearance
        else:
            grayscale_factors, gamma = self.appearance_model.get_appearance(viewpoint_camera.normalized_appearance_id)

        if self.apply_on_gaussian is True:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            with torch.no_grad():
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            override_color = torch.clamp_min(sh2rgb + 0.5, 0.0)
            grayscale_factors = grayscale_factors.reshape((1, -1))
            gamma = gamma.reshape((1, -1))
            override_color = torch.pow(override_color + 1e-5, gamma)  # +1e-5 to avoid NaN
            override_color = override_color * grayscale_factors
            outputs = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, override_color)
        else:
            outputs = super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, override_color)
            rendered_image = outputs["render"]
            outputs["render_org"] = rendered_image.clone()

            # apply appearance transform
            rendered_image = torch.pow(rendered_image, gamma)
            rendered_image = rendered_image * grayscale_factors

            # store transformed result
            outputs["render"] = rendered_image

        return outputs

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        appearance = self.appearance_config
        self.appearance_model = AppearanceModel(
            n_input_dims=1,
            n_grayscale_factors=appearance.n_grayscale_factors,
            n_gammas=appearance.n_gammas,
            n_neurons=appearance.n_neurons,
            n_hidden_layers=appearance.n_hidden_layers,
            n_frequencies=appearance.n_frequencies,
            grayscale_factors_activation=appearance.grayscale_factors_activation,
            gamma_activation=appearance.gamma_activation,
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
