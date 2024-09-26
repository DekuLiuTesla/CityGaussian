from typing import Dict, Tuple, Union, Callable, Optional, List

import lightning
import torch
import math
from .renderer import Renderer
from .vanilla_2dgs_renderer import Vanilla2DGSRenderer
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.ssim import ssim

class SepDepth2DGSRenderer(Vanilla2DGSRenderer):
    def __init__(
            self,
            depth_ratio: float = 0.,
            lambda_normal: float = 0.05,
            lambda_dist: float = 0.,
            normal_regularization_from_iter: int = 7000,
            dist_regularization_from_iter: int = 3000,
            init: float = 0.2,
            final_factor: float = 0.002,
            max_steps: int = 30_000,
            depth_loss_type: str = "l1",
    ):
        super().__init__(
            depth_ratio=depth_ratio,
            lambda_normal=lambda_normal,
            lambda_dist=lambda_dist,
            normal_regularization_from_iter=normal_regularization_from_iter,
            dist_regularization_from_iter=dist_regularization_from_iter,
        )
        self.config = {
            "depth_loss_weight": {
                "init": init,
                "final_factor": final_factor,
                "max_steps": max_steps,
            },
            "depth_loss_type": depth_loss_type,
        }

        if self.config["depth_loss_type"] == "l1":
            self._get_depth_loss = self._depth_l1_loss
        elif self.config["depth_loss_type"] == "l2":
            self._get_depth_loss = self._depth_l2_loss
        elif self.config["depth_loss_type"] == "ssim":
            self._get_depth_loss = self._depth_ssim_loss
        else:
            raise NotImplementedError()

    def train_metrics(self, pl_module, step: int, batch, outputs):
        metrics, prog_bar = pl_module.vanilla_train_metric_calculator(pl_module, step, batch, outputs)
        camera, _, gt_inverse_depth = batch

        # regularization
        lambda_normal = self.lambda_normal if step > self.normal_regularization_from_iter else 0.0
        lambda_dist = self.lambda_dist if step > self.dist_regularization_from_iter else 0.0

        rend_dist = outputs["rend_dist"]
        rend_normal = outputs['rend_normal']
        surf_normal = outputs['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()
        
        
        if gt_inverse_depth is None:
            depth_loss = torch.tensor(0., device=dist_loss.device)
        else:
            inverse_depth = 1. / (outputs["surf_depth"].clamp_min(0.).squeeze() + 1e-8)
            d_reg_weight = self.get_weight(step)
            depth_loss = self._get_depth_loss(gt_inverse_depth, inverse_depth) * d_reg_weight

        # update metrics
        if step < pl_module.hparams["gaussian"].optimization.densify_until_iter:
            
            metrics["loss"] = pl_module.lambda_dssim * (1. - metrics["ssim"])
            metrics["extra_loss"] = dist_loss + normal_loss + depth_loss + (1.0 - pl_module.lambda_dssim) * metrics["rgb_diff"]
            prog_bar["extra_loss"] = False
            
        else:
            metrics["loss"] = metrics["loss"] + dist_loss + normal_loss + depth_loss
        metrics["normal_loss"] = normal_loss
        prog_bar["normal_loss"] = False
        metrics["dist_loss"] = dist_loss
        prog_bar["dist_loss"] = False
        metrics["depth_loss"] = depth_loss
        prog_bar["depth_loss"] = False

        return metrics, prog_bar

    def get_weight(self, step: int):
        return self.config["depth_loss_weight"]["init"] * (self.config["depth_loss_weight"]["final_factor"] ** min(step / self.config["depth_loss_weight"]["max_steps"], 1))

    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_ssim_loss(self, a, b):
        from internal.utils.ssim import ssim
        return 1. - ssim(a[None], b[None])