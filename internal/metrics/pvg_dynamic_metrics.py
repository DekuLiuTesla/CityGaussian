from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch

from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class PVGDynamicMetrics(VanillaMetrics):
    velocity_reg: float = 0.001
    t_reg: float = 0.
    opacity_entropy_reg: float = 0.

    def instantiate(self, *args, **kwargs) -> "PVGDynamicMetricsModule":
        return PVGDynamicMetricsModule(self)


class PVGDynamicMetricsModule(VanillaMetricsImpl):
    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        basic_metrics, pbar = super()._get_basic_metrics(pl_module, gaussian_model, batch, outputs)

        # sparse velocity loss
        if self.config.velocity_reg > 0:
            velocity_map = outputs["average_velocity"] / outputs["alpha"].detach().clamp_min(1e-5)
            v_reg_loss = torch.abs(velocity_map).mean() * self.config.velocity_reg
            basic_metrics["loss"] = basic_metrics["loss"] + v_reg_loss
            basic_metrics["v_reg"] = v_reg_loss
            pbar["v_reg"] = True

        if self.config.t_reg > 0:
            t_reg_loss = -torch.abs(outputs["scale_t"] / outputs["alpha"].detach().clamp_min(1e-5)).mean() * self.config.t_reg
            basic_metrics["loss"] = basic_metrics["loss"] + t_reg_loss
            basic_metrics["t_reg"] = t_reg_loss
            pbar["t_reg"] = True

        if self.config.opacity_entropy_reg > 0:
            alpha = outputs["alpha"].detach()
            o = alpha.clamp(1e-6, 1 - 1e-6)
            loss_opacity_entropy = -(o * torch.log(o)).mean() * self.config.opacity_entropy_reg
            basic_metrics["loss"] = basic_metrics["loss"] + loss_opacity_entropy
            basic_metrics["o_reg"] = loss_opacity_entropy
            pbar["o_reg"] = True

        return basic_metrics, pbar
