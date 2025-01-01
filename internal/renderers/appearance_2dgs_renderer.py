from dataclasses import dataclass
from typing import List
import torch
import lightning
from gsplat import spherical_harmonics
from ..cameras import Camera
from .renderer import RendererConfig
from .vanilla_2dgs_renderer import Vanilla2DGSRenderer
from .gsplat_appearance_embedding_renderer import (
    GSplatAppearanceEmbeddingMipRendererModule,
    Model as AppearanceModel,
    ModelConfig as AppearanceModelConfig,

    OptimizationConfig as AppearanceModelOptimizationConfig,
)


@dataclass
class Appearance2DGSRenderer(RendererConfig):
    depth_ratio: float = 0.

    model: AppearanceModelConfig = AppearanceModelConfig()

    optimization: AppearanceModelOptimizationConfig = AppearanceModelOptimizationConfig()

    def instantiate(self, *args, **kwargs) -> "Appearance2DGSRendererModule":
        return Appearance2DGSRendererModule(self)


class Appearance2DGSRendererModule(Vanilla2DGSRenderer):
    def __init__(self, config: Appearance2DGSRenderer):
        super().__init__(config.depth_ratio)
        self.config = config

    def setup(self, stage: str, *args: torch.Any, **kwargs: torch.Any) -> torch.Any:
        super().setup(stage, *args, **kwargs)

        lightning_module = kwargs.get("lightning_module", None)

        if lightning_module is not None:
            if self.config.model.n_appearances <= 0:
                max_input_id = 0
                appearance_group_ids = lightning_module.trainer.datamodule.dataparser_outputs.appearance_group_ids
                if appearance_group_ids is not None:
                    for i in appearance_group_ids.values():
                        if i[0] > max_input_id:
                            max_input_id = i[0]
                n_appearances = max_input_id + 1
                self.config.model.n_appearances = n_appearances

            self._setup_model()
            print(self.model)

    def _setup_model(self, device=None):
        self.model = AppearanceModel(self.config.model)

        if device is not None:
            self.model.to(device=device)

    def load_state_dict(self, state_dict, strict: bool = True):
        self.config.model.n_appearances = state_dict["model.embedding.weight"].shape[0]
        self._setup_model(device=state_dict["model.embedding.weight"].device)
        return super().load_state_dict(state_dict, strict)

    def training_setup(self, module: lightning.LightningModule):
        embedding_optimizer, embedding_scheduler = GSplatAppearanceEmbeddingMipRendererModule._create_optimizer_and_scheduler(
            self.model.embedding.parameters(),
            "embedding",
            lr_init=self.config.optimization.embedding_lr_init,
            lr_final_factor=self.config.optimization.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.warm_up,
        )
        network_optimizer, network_scheduler = GSplatAppearanceEmbeddingMipRendererModule._create_optimizer_and_scheduler(
            self.model.network.parameters(),
            "embedding_network",
            lr_init=self.config.optimization.lr_init,
            lr_final_factor=self.config.optimization.lr_final_factor,
            max_steps=self.config.optimization.max_steps,
            eps=self.config.optimization.eps,
            warm_up=self.config.optimization.warm_up,
        )

        return [embedding_optimizer, network_optimizer], [embedding_scheduler, network_scheduler]

    def training_forward(self, step: int, module: lightning.LightningModule, viewpoint_camera: Camera, pc, bg_color: torch.Tensor, render_types: List = None, **kwargs):
        callable = self
        if step < self.config.optimization.warm_up:
            callable = super().forward
        return callable(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            render_types=render_types,
            **kwargs,
        )

    def forward(self, viewpoint_camera: Camera, pc, bg_color: torch.Tensor, scaling_modifier=1, **kwargs):
        detached_xyz = pc.get_xyz.detach()
        view_directions = detached_xyz - viewpoint_camera.camera_center  # (N, 3)
        view_directions = view_directions / view_directions.norm(dim=-1, keepdim=True)
        base_rgb = spherical_harmonics(pc.active_sh_degree, view_directions, pc.get_features) + 0.5
        rgb_offset = self.model(pc.get_appearance_features(), viewpoint_camera.appearance_id, view_directions) * 2 - 1.
        rgbs = torch.clamp(base_rgb + rgb_offset, min=0., max=1.)

        kwargs["colors_precomp"] = rgbs
        return super().forward(viewpoint_camera, pc, bg_color, scaling_modifier, **kwargs)
