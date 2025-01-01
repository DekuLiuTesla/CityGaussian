from typing import Literal, Tuple, Dict, Any
from dataclasses import dataclass, field
import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from .gs2d_metrics import GS2DMetrics, GS2DMetricsImpl


@dataclass
class WeightScheduler:
    init: float = 1.0

    final_factor: float = 0.01

    max_steps: int = 30_000


@dataclass
class CityGSV2Metrics(GS2DMetrics):
    lambda_normal: float = 0.05

    normal_regularization_from_iter: int = 7000

    depth_loss_type: Literal["l1", "l1+ssim", "l2", "kl"] = "l1"

    depth_loss_ssim_weight: float = 0.2

    depth_loss_weight: WeightScheduler = field(default_factory=lambda: WeightScheduler())

    depth_normalized: bool = False

    depth_output_key: str = "inverse_depth"

    def instantiate(self, *args, **kwargs) -> "CityGSV2MetricsModule":
        return CityGSV2MetricsModule(self)


class CityGSV2MetricsModule(GS2DMetricsImpl):
    config: CityGSV2Metrics

    def setup(self, stage: str, pl_module):
        super().setup(stage, pl_module)

        if self.config.depth_loss_type == "l1":
            self._get_inverse_depth_loss = self._depth_l1_loss
        elif self.config.depth_loss_type == "l1+ssim":
            self.depth_ssim = StructuralSimilarityIndexMeasure()
            self._get_inverse_depth_loss = self._depth_l1_and_ssim_loss
        elif self.config.depth_loss_type == "l2":
            self._get_inverse_depth_loss = self._depth_l2_loss
        # elif self.config.depth_loss_type == "kl":
        #     self._get_inverse_depth_loss = self._depth_kl_loss
        else:
            raise NotImplementedError()

    def _depth_l1_loss(self, a, b):
        return torch.abs(a - b).mean()

    def _depth_l1_and_ssim_loss(self, a, b):
        l1_loss = self._depth_l1_loss(a, b)
        ssim_metric = self.depth_ssim(a[None, None, ...], b[None, None, ...])

        return (1 - self.config.depth_loss_ssim_weight) * l1_loss + self.config.depth_loss_ssim_weight * (1 - ssim_metric)

    def _depth_l2_loss(self, a, b):
        return ((a - b) ** 2).mean()

    def _depth_kl_loss(self, a, b):
        pass

    def get_inverse_depth_metric(self, batch, outputs):
        # TODO: apply mask

        camera, _, gt_inverse_depth = batch

        if gt_inverse_depth is None:
            return torch.tensor(0., device=camera.device)
        
        predicted_inverse_depth = 1. / (outputs["surf_depth"].clamp_min(0.).squeeze() + 1e-8)
        if self.config.depth_normalized:
            # with torch.no_grad():
            clamp_val = (predicted_inverse_depth.mean() + 2 * predicted_inverse_depth.std()).item()
            predicted_inverse_depth = predicted_inverse_depth.clamp(max=clamp_val) / clamp_val
            gt_inverse_depth = gt_inverse_depth.clamp(max=clamp_val) / clamp_val

        if isinstance(gt_inverse_depth, tuple):
            gt_inverse_depth, gt_inverse_depth_mask = gt_inverse_depth

            gt_inverse_depth = gt_inverse_depth * gt_inverse_depth_mask
            predicted_inverse_depth = predicted_inverse_depth * gt_inverse_depth_mask

        return self._get_inverse_depth_loss(gt_inverse_depth, predicted_inverse_depth)

    def get_weight(self, step: int):
        return self.config.depth_loss_weight.init * (self.config.depth_loss_weight.final_factor ** min(step / self.config.depth_loss_weight.max_steps, 1))

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, pbar = super().get_train_metrics(pl_module, gaussian_model, step, batch, outputs)

        d_reg_weight = self.get_weight(step)
        d_reg = self.get_inverse_depth_metric(batch, outputs) * d_reg_weight

        metrics["d_reg"] = d_reg
        metrics["d_w"] = d_reg_weight
        pbar["d_reg"] = True
        pbar["d_w"] = True

        if step < pl_module.hparams["density"].densify_until_iter:
            pbar["extra_loss"] = False
            metrics["loss"] = pl_module.hparams["metric"].lambda_dssim * (1. - metrics["ssim"]) + metrics["dist_loss"] + metrics["normal_loss"] + d_reg
            metrics["extra_loss"] = (1.0 - pl_module.hparams["metric"].lambda_dssim) * metrics["rgb_diff"]
        else:
            metrics["loss"] = metrics["loss"] + d_reg

        return metrics, pbar

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics, pbar = super().get_validate_metrics(pl_module, gaussian_model, batch, outputs)

        d_reg = self.get_inverse_depth_metric(batch, outputs)

        metrics["loss"] = metrics["loss"] + d_reg
        metrics["d_reg"] = d_reg
        pbar["d_reg"] = True

        return metrics, pbar
