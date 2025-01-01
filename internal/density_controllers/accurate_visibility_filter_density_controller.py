"""
pip install git+https://github.com/yzslab/gsplat.git@accurate_visibility_filter

Get visibility filter from rasterization instead of projection.
This filter is more accurate, and can improve the evaluation metrics a little bit.
"""

from dataclasses import dataclass
import torch
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl


@dataclass
class AccurateVisibilityFilterDensityController(VanillaDensityController):
    def instantiate(self, *args, **kwargs) -> "AccurateVisibilityFilterDensityControllerModule":
        return AccurateVisibilityFilterDensityControllerModule(self)


class AccurateVisibilityFilterDensityControllerModule(VanillaDensityControllerImpl):
    def update_states(self, outputs):
        viewspace_point_tensor, radii = outputs["viewspace_points"], outputs["radii"]
        visibility_filter = viewspace_point_tensor.has_hit_any_pixels
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update states
        self.max_radii2D[visibility_filter] = torch.max(
            self.max_radii2D[visibility_filter],
            radii[visibility_filter]
        )
        xys_grad = viewspace_point_tensor.grad
        if self.config.absgrad is True:
            xys_grad = viewspace_point_tensor.absgrad
        self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)
