"""
3D Gaussian Splatting as Markov Chain Monte Carlo
https://ubc-vision.github.io/3dgs-mcmc/

Most codes are copied from https://github.com/ubc-vision/3dgs-mcmc
"""

from typing import Tuple, Dict, Any
import torch

from .metric import MetricImpl
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


class MCMCMetrics(VanillaMetrics):
    opacity_reg: float = 0.01

    scale_reg: float = 0.01

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return MCMCMetricsImpl(self)


class MCMCMetricsImpl(VanillaMetricsImpl):
    def reg_loss(self, gaussian_model, basic_metrics: Tuple[Dict[str, Any], Dict[str, bool]]):
        opacity_reg_loss = self.config.opacity_reg * torch.abs(gaussian_model.get_opacity).mean()
        scale_reg_loss = self.config.scale_reg * torch.abs(gaussian_model.get_scaling).mean()

        basic_metrics[0]["loss"] = basic_metrics[0]["loss"] + opacity_reg_loss + scale_reg_loss
        basic_metrics[0]["o_reg"] = opacity_reg_loss
        basic_metrics[0]["s_reg"] = scale_reg_loss

        basic_metrics[1]["o_reg"] = True
        basic_metrics[1]["s_reg"] = True

        return basic_metrics

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)
        return self.reg_loss(gaussian_model, basic_metrics)

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        basic_metrics = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)
        return self.reg_loss(gaussian_model, basic_metrics)
