from typing import Literal, Dict, Union, Optional, Tuple, List
from dataclasses import dataclass, field

import numpy as np
import torch

from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel, OptimizationConfig as VanillaGaussianOptimizationConfig
from internal.schedulers import Scheduler, ExponentialDecayScheduler


@dataclass
class OptimizationConfig(VanillaGaussianOptimizationConfig):
    appearance_feature_lr_init: float = 2e-3

    appearance_feature_lr_scheduler: Optional[Scheduler] = None


@dataclass
class AppearanceFeatureGaussian(VanillaGaussian):
    appearance_feature_dims: int = 64

    appearance_feature_init_type: Literal["zero", "rand", "normal"] = "zero"

    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig())

    def instantiate(self, *args, **kwargs) -> "AppearanceFeatureGaussianModel":
        return AppearanceFeatureGaussianModel(self)


class AppearanceFeatureGaussianModel(VanillaGaussianModel):
    config: AppearanceFeatureGaussian

    _appearance_feature_name = "appearance_features"

    def __init__(self, config: VanillaGaussian) -> None:
        super().__init__(config)
        self._names = tuple(list(self._names) + [self._appearance_feature_name])

    def create_appearance_features(self, n: int):
        appearance_features = torch.empty((n, self.config.appearance_feature_dims), dtype=torch.float)
        if self.config.appearance_feature_init_type == "zero":
            torch.nn.init.zeros_(appearance_features)
        elif self.config.appearance_feature_init_type == "rand":
            appearance_features.copy_(torch.rand_like(appearance_features))
        elif self.config.appearance_feature_init_type == "normal":
            torch.nn.init.normal_(appearance_features)

        return torch.nn.Parameter(appearance_features.requires_grad_(True))

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        property_dict[self._appearance_feature_name] = self.create_appearance_features(xyz.shape[0])

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        property_dict[self._appearance_feature_name] = self.create_appearance_features(n)

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

        appearance_feature_optimizer = torch.optim.Adam(
            [{"params": [self.gaussians[self._appearance_feature_name]], "name": self._appearance_feature_name}],
            lr=self.config.optimization.appearance_feature_lr_init,
        )
        optimizers.append(appearance_feature_optimizer)
        if self.config.optimization.appearance_feature_lr_scheduler is not None:
            appearance_feature_scheduler = self.config.optimization.appearance_feature_lr_scheduler.instantiate().get_scheduler(
                appearance_feature_optimizer,
                self.config.optimization.appearance_feature_lr_init,
            )
            if isinstance(schedulers, list) is False:
                schedulers = [schedulers]
            schedulers.append(appearance_feature_scheduler)

        return optimizers, schedulers

    def get_appearance_features(self) -> torch.Tensor:
        return self.gaussians[self._appearance_feature_name]
