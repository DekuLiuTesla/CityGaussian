from dataclasses import dataclass
from typing import Tuple, Dict, Any
import torch
import torch.nn.functional as F

from .metric import Metric, MetricImpl


@dataclass
class Feature3DGSMetrics(Metric):
    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return Feature3DGSMetricImpl(self)


class Feature3DGSMetricImpl(MetricImpl):
    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics = {}
        metrics_pbar = {}

        _, _, gt_feature_map = batch

        feature_map = outputs["features"]
        feature_map = F.interpolate(feature_map.unsqueeze(0), size=(gt_feature_map.shape[1], gt_feature_map.shape[2]), mode='bilinear', align_corners=True).squeeze(0)

        l1_loss = torch.abs((feature_map - gt_feature_map)).mean()

        metrics["loss"] = l1_loss
        metrics_pbar["loss"] = True

        return metrics, metrics_pbar
