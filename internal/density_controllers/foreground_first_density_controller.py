from typing import Tuple, Optional, Union, List, Dict
from dataclasses import dataclass, field
import os
import torch
from torch import nn
from lightning import LightningModule

from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.general_utils import build_rotation
from .density_controller import DensityController, DensityControllerImpl, Utils


@dataclass
class ForegroundFirstDensityController(DensityController):
    partition: str
    """Partition data directory"""

    partition_idx: int

    max_grad_decay_factor: float = 2

    max_radius_factor: float = 1.5
    """for those 'distance >= max_radius_factor * radius', 'grad = grad / max_grad_decay_factor'"""

    percent_dense: float = 0.01

    densification_interval: int = 100

    opacity_reset_interval: int = 3000

    densify_from_iter: int = 500

    densify_until_iter: int = 15_000

    densify_grad_threshold: float = 0.0002

    cull_opacity_threshold: float = 0.005
    """threshold of opacity for culling gaussians."""

    camera_extent_factor: float = 1.

    scene_extent_override: float = -1.

    absgrad: bool = False

    acc_vis: bool = False

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return ForegroundFirstDensityControllerModule(self)


class ForegroundFirstDensityControllerModule(DensityControllerImpl):
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        self.avoid_state_dict = {"pl": pl_module} 

        if stage == "fit":
            self.cameras_extent = pl_module.trainer.datamodule.dataparser_outputs.camera_extent * self.config.camera_extent_factor
            self.prune_extent = pl_module.trainer.datamodule.prune_extent * self.config.camera_extent_factor

            if self.config.scene_extent_override > 0:
                self.cameras_extent = self.config.scene_extent_override
                self.prune_extent = self.config.scene_extent_override
                print(f"Override scene extent with {self.config.scene_extent_override}")

            self._init_state(pl_module.gaussian_model.n_gaussians, pl_module.device)

        # load partition data
        partition_data = torch.load(os.path.join(
            self.config.partition,
            "partitions.pt",
        ))
        partition_size = partition_data["scene_config"]["partition_size"]
        self.register_buffer(
            "partition_radius",
            torch.sqrt((torch.tensor(partition_size * 0.5, dtype=torch.float) ** 2) * 2),
            persistent=False,
        )
        # get transform matrix
        try:
            rotation_transform = partition_data["extra_data"]["rotation_transform"]
        except:
            print("FFDensityController: No orientation transform")
            rotation_transform = torch.eye(4, dtype=torch.float, device=pl_module.device)
        self.register_buffer(
            "rotation_transform",
            rotation_transform,
            persistent=False,
        )
        # get bounding box
        partition_center = partition_data["partition_coordinates"]["xy"][self.config.partition_idx] + partition_size * 0.5
        self.register_buffer(
            "partition_center",
            partition_center,
            persistent=False,
        )
        # partition_bbox_min = partition_center - partition_size
        # partition_bbox_max = partition_center + partition_size
        # self.register_buffer("partition_bbox_min", partition_bbox_min, persistent=False)
        # self.register_buffer("partition_bbox_max", partition_bbox_max, persistent=False)

        print("partition_idx=#{}, id={}, transform={}, center={}, radius={}".format(
            self.config.partition_idx,
            partition_data["partition_coordinates"]["id"][self.config.partition_idx],
            self.rotation_transform.tolist(),
            self.partition_center.tolist(),
            self.partition_radius.item(),
        ))

    def log_metric(self, name, value):
        self.avoid_state_dict["pl"].logger.log_metrics(
            {
                "density/{}".format(name): value,
            },
            step=self.avoid_state_dict["pl"].trainer.global_step,
        )

    def _init_state(self, n_gaussians: int, device):
        max_radii2D = torch.zeros((n_gaussians), device=device)
        xyz_gradient_accum = torch.zeros((n_gaussians, 1), device=device)
        denom = torch.zeros((n_gaussians, 1), device=device)

        self.register_buffer("max_radii2D", max_radii2D, persistent=True)
        self.register_buffer("xyz_gradient_accum", xyz_gradient_accum, persistent=True)
        self.register_buffer("denom", denom, persistent=True)

    def before_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        outputs["viewspace_points"].retain_grad()

    def after_backward(self, outputs: dict, batch, gaussian_model: VanillaGaussianModel, optimizers: List, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return

        with torch.no_grad():
            self.update_states(outputs)

            # densify and pruning
            if global_step > self.config.densify_from_iter and global_step % self.config.densification_interval == 0:
                size_threshold = 20 if global_step > self.config.opacity_reset_interval else None
                self._densify_and_prune(
                    max_screen_size=size_threshold,
                    gaussian_model=gaussian_model,
                    optimizers=optimizers,
                )

            if global_step % self.config.opacity_reset_interval == 0 or \
                    (
                        torch.all(pl_module.background_color == 1.) and global_step == self.config.densify_from_iter
                    ):
                self._reset_opacities(gaussian_model, optimizers)

    def update_states(self, outputs):
        viewspace_point_tensor, visibility_filter, radii = outputs["viewspace_points"], outputs["visibility_filter"], outputs["radii"]
        if self.config.acc_vis:
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

    def _add_densification_stats(self, grad, update_filter, scale: Union[float, int, None]):
        scaled_grad = grad[update_filter, :2]
        if scale is not None:
            scaled_grad = scaled_grad * scale
        grad_norm = torch.norm(scaled_grad, dim=-1, keepdim=True)

        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1

    def _get_grad_decay_factors(self, gaussian_model):
        # decay grads based on distance (xy only)
        # transform 3D means
        transformed_means = gaussian_model.get_means() @ self.rotation_transform[:2, :3].T + self.rotation_transform[:2, 3]
        # calculate distances
        distance_to_partition_center = torch.norm(transformed_means - self.partition_center, dim=-1)

        distance_radius_factor = distance_to_partition_center / self.partition_radius
        distance_radius_factor_normalized = torch.clamp_max((distance_radius_factor - 1) / (self.config.max_radius_factor - 1), max=1.)
        decay_factors = (distance_radius_factor_normalized * (self.config.max_grad_decay_factor - 1)) + 1
        where_inside_partition = distance_to_partition_center <= self.partition_radius
        decay_factors[where_inside_partition] = 1.

        assert torch.all(decay_factors >= 1.)

        return decay_factors


    def _densify_and_prune(self, max_screen_size, gaussian_model: VanillaGaussianModel, optimizers: List):
        min_opacity = self.config.cull_opacity_threshold
        prune_extent = self.prune_extent

        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        grads = grads / self._get_grad_decay_factors(gaussian_model=gaussian_model).unsqueeze(-1)

        # densify
        self._densify_and_clone(grads, gaussian_model, optimizers)
        self._densify_and_split(grads, gaussian_model, optimizers)

        # prune
        prune_mask = (gaussian_model.get_opacities() < min_opacity).squeeze()
        self.log_metric("min_opacity_count", prune_mask.sum())

        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            self.log_metric("big_points_vs_count", big_points_vs.sum())
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * prune_extent
            self.log_metric("big_points_ws_count", big_points_ws.sum())
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.log_metric("prune_count", prune_mask.sum())
        self._prune_points(prune_mask, gaussian_model, optimizers)

        torch.cuda.empty_cache()

    def _densify_and_clone(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        self.log_metric("densify_and_clone_big_grad", selected_pts_mask.sum())
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

    def _split_means_and_scales(self, gaussian_model, selected_pts_mask, N):
        scales = gaussian_model.get_scales()
        device = scales.device

        stds = scales[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(gaussian_model.get_property("rotations")[selected_pts_mask]).repeat(N, 1, 1)
        # Split means and scales, they are a little bit different
        new_means = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussian_model.get_means()[selected_pts_mask].repeat(N, 1)
        new_scales = gaussian_model.scale_inverse_activation(scales[selected_pts_mask].repeat(N, 1) / (0.8 * N))

        new_properties = {
            "means": new_means,
            "scales": new_scales,
        }

        return new_properties

    def _split_properties(self, gaussian_model, selected_pts_mask, N: int):
        new_properties = self._split_means_and_scales(gaussian_model, selected_pts_mask, N)

        # Split other properties
        for key, value in gaussian_model.properties.items():
            if key in new_properties:
                continue
            new_properties[key] = value[selected_pts_mask].repeat(N, *[1 for _ in range(value[selected_pts_mask].dim() - 1)])

        return new_properties

    def _densify_and_split(self, grads, gaussian_model: VanillaGaussianModel, optimizers: List, N: int = 2):
        grad_threshold = self.config.densify_grad_threshold
        percent_dense = self.config.percent_dense
        scene_extent = self.cameras_extent

        device = gaussian_model.get_property("means").device
        n_init_points = gaussian_model.n_gaussians
        scales = gaussian_model.get_scales()

        # The number of Gaussians and `grads` is different after cloning, so padding is required
        padded_grad = torch.zeros((n_init_points,), device=device)
        padded_grad[:grads.shape[0]] = grads.squeeze()

        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        self.log_metric("densify_and_split_big_grad", selected_pts_mask.sum())
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

    def _densification_postfix(self, new_properties: Dict, gaussian_model, optimizers):
        new_parameters = Utils.cat_tensors_to_properties(new_properties, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # re-init states
        self._init_state(gaussian_model.n_gaussians, gaussian_model.get_property("means").device)

    def _prune_points(self, mask, gaussian_model: VanillaGaussianModel, optimizers: List):
        """
        Args:
            mask: `True` indicating the Gaussians to be pruned
            gaussian_model
            optimizers
        """
        valid_points_mask = ~mask  # `True` to keep
        new_parameters = Utils.prune_properties(valid_points_mask, gaussian_model, optimizers)
        gaussian_model.properties = new_parameters

        # prune states
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def _reset_opacities(self, gaussian_model: VanillaGaussianModel, optimizers: List):
        opacities_new = gaussian_model.opacity_inverse_activation(torch.min(
            gaussian_model.get_opacities(),
            torch.ones_like(gaussian_model.get_opacities()) * 0.01,
        ))
        new_parameters = Utils.replace_tensors_to_properties(tensors={
            "opacities": opacities_new,
        }, optimizers=optimizers)
        gaussian_model.update_properties(new_parameters)

    def on_load_checkpoint(self, module, checkpoint):
        self._init_state(checkpoint["state_dict"]["density_controller.max_radii2D"].shape[0], module.device)

    def after_density_changed(self, gaussian_model, optimizers: List, pl_module: LightningModule) -> None:
        self._init_state(gaussian_model.n_gaussians, pl_module.device)
