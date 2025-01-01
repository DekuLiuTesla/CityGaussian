from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union, List

import torch
from internal.schedulers import Scheduler, ExponentialDecayScheduler
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel, OptimizationConfig as VanillaOptimizationConfig


@dataclass
class OptimizationConfig(VanillaOptimizationConfig):
    t_lr_init: float = 0.0008
    t_lr_scheduler: Scheduler = field(default_factory=lambda: {
        "class_path": "ExponentialDecayScheduler",
        "init_args": {
            "lr_final": 0.000008,
            "max_steps": 30_000,
        },
    })

    scale_t_lr: float = 0.002

    velocity_lr: float = 0.001


@dataclass
class PeriodicVibrationGaussian(VanillaGaussian):
    t_init: float = 0.2

    time_duration: Tuple[float, float] = (-0.5, 0.5)

    cycle: float = 0.2

    velocity_decay: float = 1.0

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "PeriodicVibrationGaussianModel":
        return PeriodicVibrationGaussianModel(self)


class PeriodicVibrationGaussianModel(VanillaGaussianModel):
    config: PeriodicVibrationGaussian

    def get_extra_property_names(self):
        return [
            "t",  # [N, 1]; life peak, denoted as τ, which represents the point’s moment of maximum prominence over time
            "scale_t",  # [N, 1]; opacity decaying β, governs the lifespan around τ, without activation
            "velocity",  # [N, 3]; vibrating direction and denotes the instant velocity at time τ
        ]

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        fused_times = (torch.rand((xyz.shape[0], 1)) * 1.2 - 0.1) * (self.config.time_duration[1] - self.config.time_duration[0]) + self.config.time_duration[0]
        # fused_times = torch.zeros((xyz.shape[0], 1), dtype=torch.float)

        dist_t = torch.full_like(fused_times, (self.config.time_duration[1] - self.config.time_duration[0]) * self.config.t_init)
        scales_t = self.scale_t_inverse_activation(torch.sqrt(dist_t))

        velocity = torch.full((xyz.shape[0], 3), 0.)

        property_dict["t"] = fused_times
        property_dict["scale_t"] = scales_t
        property_dict["velocity"] = velocity

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        property_dict["t"] = torch.empty((n, 1), dtype=torch.float)
        property_dict["scale_t"] = torch.empty((n, 1), dtype=torch.float)
        property_dict["velocity"] = torch.empty((n, 3), dtype=torch.float)

    def training_setup(self, module: "lightning.LightningModule") -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        optimizers, schedulers = super().training_setup(module)

        t_optimizer = torch.optim.Adam(
            params=[{"name": "t", "params": [self.gaussians["t"]]}],
            lr=self.config.optimization.t_lr_init,
            eps=1e-15,
        )
        t_scheduler = self.config.optimization.t_lr_scheduler.instantiate().get_scheduler(t_optimizer, lr_init=self.config.optimization.t_lr_init)

        scale_t_and_velocity_optimizer = torch.optim.Adam(
            params=[
                {"name": "scale_t", "params": [self.gaussians["scale_t"]], "lr": self.config.optimization.scale_t_lr},
                {"name": "velocity", "params": [self.gaussians["velocity"]], "lr": self.config.optimization.velocity_lr * self.config.optimization.spatial_lr_scale},
            ],
            lr=0.,
            eps=1e-15,
        )

        optimizers += [t_optimizer, scale_t_and_velocity_optimizer]
        schedulers += [t_scheduler]

        return optimizers, schedulers

    def get_t(self):
        return self.gaussians["t"]

    def scale_t_activation(self, v):
        return torch.exp(v)

    def scale_t_inverse_activation(self, v):
        return torch.log(v)

    def get_scale_t(self):
        return self.scale_t_activation(self.gaussians["scale_t"])

    def get_velocity(self):
        return self.gaussians["velocity"]

    def get_mean_SHM(self, t):
        # vibrating motion; e.q. #6
        a = 1 / self.config.cycle * torch.pi * 2
        return self.get_means() + self.get_velocity() * torch.sin((t - self.get_t()) * a) / a

    def get_marginal_t(self, timestamp):
        # the factor of vibrating opacity; e.q. #7
        return torch.exp(-0.5 * (self.get_t() - timestamp) ** 2 / self.get_scale_t() ** 2)

    def get_average_velocity(self):
        # e.q. #10
        # ρ = β / l = get_scale_t() / self.config.cycle
        return self.get_velocity() * torch.exp(-self.get_scale_t() / self.config.cycle / 2 * self.config.velocity_decay)
