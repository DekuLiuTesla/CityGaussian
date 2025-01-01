from dataclasses import dataclass, field
from typing import Tuple, Optional, Union, List, Any

import lightning
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from gsplat.sh_decomposed import spherical_harmonics_decomposed
from internal.cameras import Camera
from . import Renderer
from .gsplat_distributed_renderer import GSplatDistributedRenderer, GSplatDistributedRendererImpl
from .gsplat_appearance_embedding_renderer import Model as AppearanceEmbeddingModel, ModelConfig, OptimizationConfig, GSplatAppearanceEmbeddingRendererModule as GSplatAppearanceEmbeddingRenderer


class GSplatDistributedAppearanceEmbeddingRendererImpl(GSplatDistributedRendererImpl):
    config: "GSplatDistributedAppearanceEmbeddingRenderer"

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        super().setup(stage, *args, **kwargs)

        self.appearance_model = AppearanceEmbeddingModel(self.config.appearance)

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        super().training_setup(module)

        # check `n_appearances`
        max_input_id = 0
        appearance_group_ids = module.trainer.datamodule.dataparser_outputs.appearance_group_ids
        if appearance_group_ids is not None:
            for i in appearance_group_ids.values():
                if i[0] > max_input_id:
                    max_input_id = i[0]
        n_appearances = max_input_id + 1
        assert self.config.appearance.n_appearances >= n_appearances, "`n_appearances` must be >= {}".format(n_appearances)

        print(self.appearance_model)
        self.appearance_model = DDP(self.appearance_model, device_ids=[module.device.index])

        embedding_optimizer, embedding_scheduler = GSplatAppearanceEmbeddingRenderer._create_optimizer_and_scheduler(
            self.appearance_model.module.embedding.parameters(),
            "embedding",
            lr_init=self.config.appearance_optimization.embedding_lr_init,
            lr_final_factor=self.config.appearance_optimization.lr_final_factor,
            max_steps=self.config.appearance_optimization.max_steps,
            eps=self.config.appearance_optimization.eps,
            warm_up=self.config.appearance_optimization.warm_up,
        )
        network_optimizer, network_scheduler = GSplatAppearanceEmbeddingRenderer._create_optimizer_and_scheduler(
            self.appearance_model.module.network.parameters(),
            "embedding_network",
            lr_init=self.config.appearance_optimization.lr_init,
            lr_final_factor=self.config.appearance_optimization.lr_final_factor,
            max_steps=self.config.appearance_optimization.max_steps,
            eps=self.config.appearance_optimization.eps,
            warm_up=self.config.appearance_optimization.warm_up,
        )

        return [embedding_optimizer, network_optimizer], [embedding_scheduler, network_scheduler]

    def get_rgbs(self, pc, camera: Camera, projection_results) -> torch.Tensor:
        is_gaussian_visible = projection_results[-1]

        detached_xyz = pc.get_xyz.detach()
        view_directions = detached_xyz[is_gaussian_visible] - camera.camera_center  # (N, 3)
        view_directions = torch.nn.functional.normalize(view_directions, dim=-1)
        base_rgb = spherical_harmonics_decomposed(
            pc.active_sh_degree,
            view_directions,
            dc=pc.get_shs_dc()[is_gaussian_visible],
            coeffs=pc.get_shs_rest()[is_gaussian_visible],
        ) + 0.5
        rgb_offset = self.appearance_model(pc.get_appearance_features()[is_gaussian_visible], camera.appearance_id, view_directions) * 2 - 1.
        rgbs = torch.zeros((is_gaussian_visible.shape[0], 3), dtype=projection_results[1].dtype, device=is_gaussian_visible.device)
        rgbs[is_gaussian_visible] = torch.clamp(base_rgb + rgb_offset, min=0., max=1.)

        return rgbs


@dataclass
class GSplatDistributedAppearanceEmbeddingRenderer(GSplatDistributedRenderer):
    appearance: ModelConfig = field(default_factory=lambda: ModelConfig())

    appearance_optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> Renderer:
        # TODO: enable warm up
        assert self.appearance_optimization.warm_up == 0, "`warm_up` must be `0` currently"
        return GSplatDistributedAppearanceEmbeddingRendererImpl(self)


# With Mip
@dataclass
class GSplatDistributedAppearanceMipRenderer(GSplatDistributedAppearanceEmbeddingRenderer):
    filter_2d_kernel_size: float = 0.1

    def instantiate(self, *args, **kwargs) -> "GSplatDistributedAppearanceMipRendererModule":
        assert self.appearance_optimization.warm_up == 0, "`warm_up` must be `0` currently"
        return GSplatDistributedAppearanceMipRendererModule(self)


class GSplatDistributedAppearanceMipRendererModule(GSplatDistributedAppearanceEmbeddingRendererImpl):
    def get_scales_and_opacities(self, pc):
        opacities, scales = pc.get_3d_filtered_scales_and_opacities()
        return scales, opacities
