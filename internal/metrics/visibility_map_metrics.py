from typing import Tuple, Dict, Any
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


class VisibilityMapMetrics(VanillaMetrics):
    vis_reg_factor: float = 0.2

    def instantiate(self, *args, **kwargs):
        return VisibilityMapMetricsImpl(self)


class VisibilityMapMetricsImpl(VanillaMetricsImpl):
    def get_train_metrics(self, pl_module, gaussian_model, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        camera, image_info, extra_data = batch
        image_name, gt_image, masked_pixels = image_info
        image = outputs["render"]

        visibility_map = outputs["visibility"]
        vis_masked_image = image * visibility_map
        vis_masked_gt_image = gt_image * visibility_map

        metrics, pbar = super().get_train_metrics(
            pl_module,
            gaussian_model,
            step,
            (camera, (image_name, vis_masked_gt_image, masked_pixels), extra_data),
            {
                "render": vis_masked_image,
            },
        )

        vis_reg = ((1. - visibility_map) ** 2).mean() * self.config.vis_reg_factor

        metrics["loss"] = metrics["loss"] + vis_reg
        metrics["vis_reg"] = vis_reg
        pbar["vis_reg"] = True

        return metrics, pbar
