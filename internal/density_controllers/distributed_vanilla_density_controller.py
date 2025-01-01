from typing import Tuple, Optional, Union, List
from dataclasses import dataclass
import torch
from lightning import LightningModule

from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl


@dataclass
class DistributedVanillaDensityController(VanillaDensityController):
    def instantiate(self, *args, **kwargs) -> "DistributedVanillaDensityControllerImpl":
        return DistributedVanillaDensityControllerImpl(self)


class DistributedVanillaDensityControllerImpl(VanillaDensityControllerImpl):
    def before_backward(self, outputs: dict, batch, gaussian_model, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        for i in outputs["projection_results_list"]:
            i[1].retain_grad()

    def update_states(self, outputs):
        cameras = outputs["cameras"]
        projection_results_list = outputs["projection_results_list"]
        visible_mask_list = outputs["visible_mask_list"]
        # processing for each projection results
        for i in range(len(projection_results_list)):
            # retrieve data
            camera = cameras[i]
            radii, xys = projection_results_list[i][0], projection_results_list[i][1]

            viewspace_point_tensor = xys
            visibility_filter = visible_mask_list[i]
            viewspace_points_grad_scale = torch.ones((2,), dtype=torch.float, device=xys.device)
            if outputs["xys_grad_scale_required"] is True:
                viewspace_points_grad_scale = 0.5 * torch.tensor([[camera.width, camera.height]], dtype=torch.float, device=xys.device)

            # update states
            self.max_radii2D[visibility_filter] = torch.max(
                self.max_radii2D[visibility_filter],
                radii[visibility_filter]
            )
            xys_grad = viewspace_point_tensor.grad
            if self.config.absgrad is True:
                xys_grad = viewspace_point_tensor.absgrad
            self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)
