"""
From Hierarchical 3D Gaussian
"""

from dataclasses import dataclass
from typing import List, Union
import torch

from internal.density_controllers.density_controller import DensityControllerImpl
from internal.models.vanilla_gaussian import VanillaGaussianModel
from .vanilla_density_controller import VanillaDensityController, VanillaDensityControllerImpl
from .logger_mixin import LoggerMixin


@dataclass
class H3DGSDensityController(VanillaDensityController):
    percent_dense: float = 0.0001

    densification_interval: int = 300

    densify_grad_threshold: float = 0.015
    """Increase this to avoid OOM"""

    def instantiate(self, *args, **kwargs) -> "H3DGSDensityControllerModule":
        return H3DGSDensityControllerModule(self)


class H3DGSDensityControllerModule(LoggerMixin, VanillaDensityControllerImpl):
    def _densify_and_prune(self, max_screen_size, gaussian_model: VanillaGaussianModel, optimizers: List):
        min_opacity = self.config.cull_opacity_threshold

        grads = self.xyz_gradient_accum
        grads[grads.isnan()] = 0.0

        max_radii2D = self.max_radii2D

        # densify
        self._densify_and_clone(grads, max_radii2D, gaussian_model, optimizers)
        self._densify_and_split(grads, max_radii2D, gaussian_model, optimizers)

        # prune
        prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
        self.log_metric("prune_count", prune_mask.sum())
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, max_radii2D, gaussian_model: VanillaGaussianModel, optimizers: List):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) * max_radii2D * torch.pow(gaussian_model.get_opacities().flatten(), 1/5.0) >= grad_threshold, True, False)
        self.log_metric("densify_and_clone_big_grad_count", selected_pts_mask.sum())
        selected_pts_mask = torch.logical_and(selected_pts_mask, gaussian_model.get_opacities().flatten() > 0.15)
        self.log_metric("densify_and_clone_big_grad_and_opacity_count", selected_pts_mask.sum())
        # Exclude big Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(gaussian_model.get_scales(), dim=1).values <= percent_dense * scene_extent,
        )
        self.log_metric("densify_and_clone_count", selected_pts_mask.sum())

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

    def _densify_and_split(self, grads, max_radii2D, gaussian_model: VanillaGaussianModel, optimizers: List, N: int = 2):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        padded_max_radii2D = torch.zeros((n_init_points,), device=device)
        padded_max_radii2D[:max_radii2D.shape[0]] = max_radii2D

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad * padded_max_radii2D * torch.pow(gaussian_model.get_opacities().flatten(), 1/5.0) >= grad_threshold, True, False)
        self.log_metric("densify_and_split_big_grad_count", selected_pts_mask.sum())
        selected_pts_mask = torch.logical_and(selected_pts_mask, gaussian_model.get_opacities().flatten() > 0.15)
        self.log_metric("densify_and_split_big_grad_and_opacity_count", selected_pts_mask.sum())
        # Exclude small Gaussians
        selected_pts_mask = torch.logical_and(
            selected_pts_mask,
            torch.max(
                scales,
                dim=1,
            ).values > percent_dense * scene_extent,
        )
        self.log_metric("densify_and_split_count", selected_pts_mask.sum())

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * selected_pts_mask.sum(),
                device=device,
                dtype=torch.bool,
            ),
        ))
        self._prune_points(prune_filter, gaussian_model, optimizers)

    def _add_densification_stats(self, grad, update_filter, scale: Union[float, int, None]):
        scaled_grad = grad[update_filter, :2]
        if scale is not None:
            scaled_grad = scaled_grad * scale
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)

        # use maximum grad values
        self.xyz_gradient_accum[update_filter] = torch.max(grad_norm, self.xyz_gradient_accum[update_filter])
