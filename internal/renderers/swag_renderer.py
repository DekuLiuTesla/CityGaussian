from .renderer import *
from .vanilla_renderer import VanillaRenderer
import torch
from torch.distributions.uniform import Uniform
from internal.models import swag_model
from internal.utils.sh_utils import eval_gaussian_model_sh


class SWAGRenderer(Renderer):
    def __init__(
            self,
            network: swag_model.NetworkConfig = swag_model.NetworkConfig(),
            grid_encoding: swag_model.GridEncodingConfig = swag_model.GridEncodingConfig(),
            embedding: swag_model.EmbeddingConfig = swag_model.EmbeddingConfig(),
            optimization: swag_model.GridEncodingOptimizationConfig = swag_model.GridEncodingOptimizationConfig(),
            temperature: float = 0.1,
            eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.network_config = network
        self.grid_encoding_config = grid_encoding
        self.embedding_config = embedding
        self.optimization_config = optimization
        self.temperature = temperature
        self.eps = eps

    def _get_normalized_xyz(self, gaussian_model):
        xyz = gaussian_model.get_xyz.detach()
        normalized_xyz = (xyz - self.bbox_min.to(xyz.device)) / self.bbox_size.to(xyz.device)
        return normalized_xyz

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            u=None,
            **kwargs,
    ):
        colors = eval_gaussian_model_sh(viewpoint_camera, pc)  # [n, 3]
        image_conditioned_colors, image_conditioned_delta_alpha = self.swag_model(colors, self._get_normalized_xyz(gaussian_model=pc), viewpoint_camera.appearance_id)

        # fix U at 0.5 during the evaluation
        if u is None:
            u = torch.tensor(0.5, dtype=torch.float, device=bg_color.device)

        image_dependent_opacity_variation = torch.nn.functional.sigmoid(1 / self.temperature * (
                torch.log(torch.abs(image_conditioned_delta_alpha) + self.eps) +
                torch.log(u + self.eps) -
                torch.log(1 - u + self.eps)
        ))  # [n]
        final_opacity = torch.clamp_min(pc.get_opacity.squeeze(-1) - image_dependent_opacity_variation, 0).unsqueeze(-1)  # [n, 1]
        
        # inference stage use no appearance embedding
        return VanillaRenderer.render(
            means3D=pc.get_xyz,
            opacity=final_opacity,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            features=None,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            colors_precomp=colors,
        )

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            u=None,
            **kwargs,
    ):
        colors = eval_gaussian_model_sh(viewpoint_camera, pc)  # [n, 3]
        image_conditioned_colors, image_conditioned_delta_alpha = self.swag_model(colors, self._get_normalized_xyz(gaussian_model=pc), viewpoint_camera.appearance_id)

        # fix U at 0.5 during the evaluation
        if u is None:
            u = torch.tensor(0.5, dtype=torch.float, device=bg_color.device)

        image_dependent_opacity_variation = torch.nn.functional.sigmoid(1 / self.temperature * (
                torch.log(torch.abs(image_conditioned_delta_alpha) + self.eps) +
                torch.log(u + self.eps) -
                torch.log(1 - u + self.eps)
        ))  # [n]
        final_opacity = torch.clamp_min(pc.get_opacity.squeeze(-1) - image_dependent_opacity_variation, 0).unsqueeze(-1)  # [n, 1]

        outputs = VanillaRenderer.render(
            means3D=pc.get_xyz,
            opacity=final_opacity,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            features=None,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            colors_precomp=image_conditioned_colors,
        )

        outputs["render_org"] = VanillaRenderer.render(
            means3D=pc.get_xyz,
            opacity=final_opacity,
            scales=pc.get_scaling,
            rotations=pc.get_rotation,
            features=None,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            colors_precomp=colors,
        )["render"]

        return outputs

    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        with torch.no_grad():
            # find scene size
            xyz = torch.tensor(lightning_module.trainer.datamodule.dataparser_outputs.point_cloud.xyz)
            self.bbox_min = torch.min(xyz, dim=0).values
            self.bbox_max = torch.max(xyz, dim=0).values
            self.bbox_size = (self.bbox_max - self.bbox_min) * 1.1

            print("bbox_size={}".format(self.bbox_size.cpu().numpy()))

        self.swag_model = swag_model.SWAGModel(
            network=self.network_config,
            grid_encoding=self.grid_encoding_config,
            embedding=self.embedding_config,
        )

    def training_setup(self, module) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        params = list(self.swag_model.parameters())

        self.uniform_sampler = Uniform(
            torch.tensor(0, dtype=torch.float, device=params[0].device),
            torch.tensor(1, dtype=torch.float, device=params[0].device),
        )

        # TODO: setup different optimizer and scheduler for grid and embedding
        optimizer = torch.optim.Adam(
            params=[
                {"params": params, "name": "swag_model"},
            ],
            lr=self.optimization_config.lr,
        )
        return optimizer, torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.optimization_config.lr_final_factor ** min(iter / self.optimization_config.max_steps, 1),
            verbose=False,
        )
