from dataclasses import dataclass
from typing import Tuple, Dict, Literal, Any
import torch
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from internal.utils.ssim import ssim

from .metric import Metric, MetricImpl
from ..configs.instantiate_config import InstantiatableConfig

@dataclass
class VanillaMetrics(Metric):
    lambda_dssim: float = 0.2

    rgb_diff_loss: Literal["l1", "l2"] = "l1"

    lpips_net_type: Literal["vgg", "alex", "squeeze"] = "alex"
    """
    the vanilla 3DGS uses 'vgg', but 'alex' is faster
    """

    fused_ssim: bool = False

    def instantiate(self, *args, **kwargs) -> MetricImpl:
        return VanillaMetricsImpl(self)


class VanillaMetricsImpl(MetricImpl):
    def __init__(self, config: InstantiatableConfig, *args, **kwargs) -> None:
        super().__init__(config, *args, **kwargs)

        self.no_state_dict_models = {}

    @staticmethod
    def _create_fused_ssim_adapter():
        from fused_ssim import fused_ssim
        def adapter(pred, gt):
            return fused_ssim(pred.unsqueeze(0), gt.unsqueeze(0))
        return adapter


    def setup(self, stage: str, pl_module):
        self.psnr = PeakSignalNoiseRatio()
        self.no_state_dict_models["lpips"] = LearnedPerceptualImagePatchSimilarity(normalize=True, net_type=self.config.lpips_net_type)

        self.lambda_dssim = self.config.lambda_dssim
        self.rgb_diff_loss_fn = self._l1_loss
        if self.config.rgb_diff_loss == "l2":
            print("Use L2 loss")
            self.rgb_diff_loss_fn = self._l2_loss

        self.ssim = ssim
        if self.config.fused_ssim:
            print("Fused SSIM enabled")
            self.ssim = self._create_fused_ssim_adapter()

    def _get_basic_metrics(self, pl_module, gaussian_model, batch, outputs):
        camera, image_info, _ = batch
        image_name, gt_image, masked_pixels = image_info
        image = outputs["render"]

        # calculate loss
        if masked_pixels is not None:
            gt_image = gt_image.clone()
            gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.
        rgb_diff_loss = self.rgb_diff_loss_fn(outputs["render"], gt_image)
        ssim_metric = self.ssim(outputs["render"], gt_image)
        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1. - ssim_metric)

        return {
            "loss": loss,
            "rgb_diff": rgb_diff_loss,
            "ssim": ssim_metric,
        }, {
            "loss": True,
            "rgb_diff": True,
            "ssim": True,
        }

    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        return self._get_basic_metrics(
            pl_module=pl_module,
            gaussian_model=gaussian_model,
            batch=batch,
            outputs=outputs,
        )

    def get_validate_metrics(self, pl_module, gaussian_model, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        metrics, prog_bar = self._get_basic_metrics(pl_module, gaussian_model, batch, outputs)

        camera, image_info, _ = batch
        image_name, gt_image, _ = image_info

        metrics["psnr"] = self.psnr(outputs["render"], gt_image)
        prog_bar["psnr"] = True
        metrics["lpips"] = self.no_state_dict_models["lpips"](outputs["render"].clamp(0., 1.).unsqueeze(0), gt_image.unsqueeze(0))
        prog_bar["lpips"] = True

        prog_bar["ssim"] = True

        return metrics, prog_bar

    def on_parameter_move(self, *args, **kwargs):
        if "lpips" in self.no_state_dict_models:
            self.no_state_dict_models["lpips"] = self.no_state_dict_models["lpips"].to(*args, **kwargs)

    @staticmethod
    def _l1_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.abs(predict - gt).mean()

    @staticmethod
    def _l2_loss(predict: torch.Tensor, gt: torch.Tensor):
        return torch.mean((predict - gt) ** 2)
