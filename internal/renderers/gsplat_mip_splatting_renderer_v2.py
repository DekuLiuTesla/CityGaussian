from dataclasses import dataclass

import torch

from .gsplat_v1_renderer import GSplatV1Renderer, GSplatV1RendererModule
from ..models.mip_splatting import MipSplattingModel


@dataclass
class GSplatMipSplattingRendererV2(GSplatV1Renderer):
    filter_2d_kernel_size: float = 0.1

    def instantiate(self, *args, **kwargs) -> "GSplatMipSplattingRendererV2Module":
        return GSplatMipSplattingRendererV2Module(self)


class MipSplattingRendererMixin:
    def get_scales(self, camera, gaussian_model: MipSplattingModel, **kwargs):
        opacities, scales = gaussian_model.get_3d_filtered_scales_and_opacities()

        return scales, opacities.squeeze(-1)

    def get_opacities(self, camera, gaussian_model: MipSplattingModel, projections, visibility_filter, status: torch.Any, **kwargs):
        return status, None


class GSplatMipSplattingRendererV2Module(MipSplattingRendererMixin, GSplatV1RendererModule):
    pass
