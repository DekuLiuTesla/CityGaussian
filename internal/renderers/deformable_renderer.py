from dataclasses import dataclass
from typing import Tuple, Optional, Any

import lightning
import torch
from .renderer import Renderer
from .vanilla_renderer import VanillaRenderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel
from ..models.deform_model import DeformModel
from ..utils.network_factory import NetworkFactory
from ..utils.general_utils import get_linear_noise_func
from ..utils.rigid_utils import from_homogenous, to_homogenous
from ..utils.rotation import qvec2rot
from ..utils.gaussian_utils import GaussianTransformUtils


@dataclass
class DeformNetworkConfig:
    """
    Args:
        tcnn: whether use tiny-cuda-nn as network implementation
    """

    tcnn: bool = True
    n_layers: int = 8
    n_neurons: int = 256
    is_6dof: bool = False
    rotate_xyz: bool = False
    chunk: int = -1  # avoid CUDA oom


@dataclass
class XYZEncodingConfig:
    n_frequencies: int = 10


@dataclass
class TimeEncodingConfig:
    n_frequencies: int = 6
    n_layers: int = 0
    n_neurons: int = 0
    n_output_dim: int = 30


@dataclass
class DeformableRendererOptimizationConfig:
    lr: float = 0.0008
    max_steps: int = 40_000
    lr_final_factor: float = 0.002
    eps: float = 1e-15
    warm_up: int = 3_000
    enable_ast: bool = True


class DeformableRenderer(Renderer):
    def __init__(
            self,
            deform_network: DeformNetworkConfig,
            xyz_encoding: XYZEncodingConfig,
            time_encoding: TimeEncodingConfig,
            optimization: DeformableRendererOptimizationConfig,
    ) -> None:
        super().__init__()

        self.deform_network_config = deform_network
        self.xyz_encoding_config = xyz_encoding
        self.time_encoding_config = time_encoding
        self.optimization_config = optimization

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

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        d_xyz, d_rotation, d_scaling = 0.0, 0.0, 0.0
        if step >= self.optimization_config.warm_up:
            N = pc.get_xyz.shape[0]
            time_input = viewpoint_camera.time.unsqueeze(0).expand(N, -1)
            ast_noise = 0
            if self.optimization_config.enable_ast is True:
                time_interval = 1 / ((step % self.train_set_length) + 1)
                ast_noise = torch.randn(1, 1, device=pc.get_xyz.device).expand(N, -1) * time_interval * self.smooth_term(step)
            d_xyz, d_rotation, d_scaling = self.deform_model(pc.get_xyz.detach(), time_input + ast_noise)
            torch.cuda.empty_cache()  # avoid CUDA OOM

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
        if self.deform_network_config.rotate_xyz is True:
            if torch.is_tensor(d_xyz) is True:
                normalized_qvec = torch.nn.functional.normalize(d_rotation)
                # rotate gaussians
                rotations = GaussianTransformUtils.quat_multiply(pc.get_rotation, normalized_qvec)
                # transform xyz
                so3 = qvec2rot(normalized_qvec)
                means3D = torch.matmul(pc.get_xyz.unsqueeze(1), torch.transpose(so3, 1, 2)).squeeze(1) + d_xyz
            else:
                # in warm up
                means3D = pc.get_xyz
                rotations = pc.get_rotation
        else:
            # original processing
            if self.deform_network_config.is_6dof is True:
                if torch.is_tensor(d_xyz) is False:
                    means3D = pc.get_xyz
                else:
                    means3D = from_homogenous(torch.bmm(d_xyz, to_homogenous(pc.get_xyz).unsqueeze(-1)).squeeze(-1))
            else:
                means3D = pc.get_xyz + d_xyz
            rotations = pc.get_rotation + d_rotation

        opacity = pc.get_opacity
        scales = pc.get_scaling + d_scaling
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

    def setup(self, stage: str, lightning_module, *args: Any, **kwargs: Any) -> Any:
        if stage == "fit":
            self.train_set_length = len(lightning_module.trainer.datamodule.dataparser_outputs.train_set)

        network_factory = NetworkFactory(tcnn=self.deform_network_config.tcnn)

        self.deform_model = DeformModel(
            network_factory=network_factory,
            D=self.deform_network_config.n_layers,
            W=self.deform_network_config.n_neurons,
            input_ch=3,
            multires=self.xyz_encoding_config.n_frequencies,
            t_D=self.time_encoding_config.n_layers,
            t_W=self.time_encoding_config.n_neurons,
            t_multires=self.time_encoding_config.n_frequencies,
            t_output_ch=self.time_encoding_config.n_output_dim,
            is_6dof=self.deform_network_config.is_6dof,
            chunk=self.deform_network_config.chunk,
        )
        self.smooth_term = get_linear_noise_func(lr_init=0.1, lr_final=1e-15, lr_delay_mult=0.01, max_steps=20000)

    def training_setup(self, module) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        optimizer = torch.optim.Adam(
            [{
                "params": list(self.deform_model.parameters()),
                "name": "deform",
            }],
            lr=self.optimization_config.lr,
            eps=self.optimization_config.eps,
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda iter: self.optimization_config.lr_final_factor ** min(iter / self.optimization_config.max_steps, 1),
            verbose=False,
        )

        return optimizer, scheduler
