"""
Most copied from https://github.com/humansensinglab/taming-3dgs
"""

from typing import Literal, List
from dataclasses import dataclass
import torch
from lightning import LightningModule
from PIL import ImageFilter
import torchvision.transforms as transforms
from fused_ssim import fused_ssim
from internal.renderers.gsplat_v1_renderer import GSplatV1
from gsplat.rasterize_to_weights import rasterize_to_weights
from .density_controller import DensityController
from .vanilla_density_controller import VanillaDensityControllerImpl


@dataclass
class ScoreCoefficients:
    view_importance: float = 50
    edge_importance: float = 50
    mse_importance: float = 50
    grad_importance: float = 25
    dist_importance: float = 50
    opac_importance: float = 100
    dept_importance: float = 5
    loss_importance: float = 10
    radii_importance: float = 10
    scale_importance: float = 25
    count_importance: float = 0.1
    blend_importance: float = 50


@dataclass
class Taming3DGSDensityController(DensityController):
    percent_dense: float = 0.01

    densification_interval: int = 500

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

    # Taming3DGS hyperparameters

    n_sample_cameras: int = 10

    mode: Literal["multiplier", "final_count"] = "multiplier"

    start_count: int = -1

    budget: float = 20

    cull_opacity_until: int = 27

    score_coeffs: ScoreCoefficients = ScoreCoefficients()

    def instantiate(self, *args, **kwargs) -> "Taming3DGSDensityControllerModule":
        return Taming3DGSDensityControllerModule(self)


class Taming3DGSDensityControllerModule(VanillaDensityControllerImpl):
    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        self.register_buffer("_densify_iter_num", torch.tensor(1, dtype=torch.int), persistent=True)

        if stage == "fit":
            start_count = self.config.start_count
            if start_count <= 0:
                start_count = pl_module.trainer.datamodule.dataparser_outputs.point_cloud.xyz.shape[0]
            self.counts_array = Taming3DGSUtils.get_count_array(
                start_count=start_count,
                multiplier=self.config.budget,
                densify_until_iter=self.config.densify_until_iter,
                densify_from_iter=self.config.densify_from_iter,
                densification_interval=self.config.densification_interval,
                mode=self.config.mode,
            )
            print(self.counts_array)

            pl_module.on_train_start_hooks.append(self.on_train_start)

            self.avoid_state_dict = (pl_module,)

    def on_train_start(self, gaussian_model, pl_module):
        assert pl_module.trainer.train_dataloader.max_cache_num < 0

        from tqdm.auto import tqdm

        train_set = pl_module.trainer.train_dataloader.cached

        all_edges = []
        for item in tqdm(train_set, leave=False, desc="Getting edges..."):
            image = item[1][1]
            if image.dtype == torch.uint8:
                image = image.float() / 255.
            edges_loss = Taming3DGSUtils.get_edges(image).squeeze(0)
            edges_loss_norm = (edges_loss - torch.min(edges_loss)) / (torch.max(edges_loss) - torch.min(edges_loss))
            all_edges.append(edges_loss_norm.cpu())
        self.all_edges = all_edges

    @property
    def densify_iter_num(self) -> int:
        return self._densify_iter_num.item()

    @densify_iter_num.setter
    def densify_iter_num(self, v):
        self._densify_iter_num.fill_(v)

    def _densify_and_prune(self, max_screen_size, gaussian_model, optimizers: List):
        # calculate mean grads
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        grads = grads.squeeze(-1)

        pl_module = self.avoid_state_dict[0]

        # sample cameras
        n_cameras = len(pl_module.trainer.train_dataloader.cached)
        sample_cameras = []
        sample_edge_losses = []
        for camera_idx in torch.randperm(n_cameras, dtype=torch.int)[:self.config.n_sample_cameras].tolist():
            sample_cameras.append(pl_module.trainer.train_dataloader.cached[camera_idx])
            sample_edge_losses.append(self.all_edges[camera_idx])

        # calculate importance
        gaussian_importance = Taming3DGSUtils.compute_gaussian_score(
            gaussian_model=gaussian_model,
            renderer=pl_module.renderer,
            global_steps=pl_module.global_step,
            all_grads=grads,
            sample_cameras=sample_cameras,
            edge_losses=sample_edge_losses,
            bg_color=pl_module._fixed_background_color().to(device=pl_module.device),
            score_coeffs=self.config.score_coeffs,
            lambda_dssim=pl_module.metric.config.lambda_dssim,
        )

        self._densify_and_prune_with_scores(
            grads=grads,
            scores=gaussian_importance,
            gaussian_model=gaussian_model,
            optimizers=optimizers,
        )

        # prune
        self._opacity_culling(gaussian_importance, max_screen_size, gaussian_model, optimizers)

        self.densify_iter_num = self.densify_iter_num + 1

        torch.cuda.empty_cache()

    def _densify_and_prune_with_scores(
            self,
            grads,
            scores,
            gaussian_model,
            optimizers,
    ):
        extent = self.cameras_extent
        scales = gaussian_model.get_scales()
        max_scales = torch.max(scales, dim=-1).values

        grad_qualifiers = grads >= self.config.densify_grad_threshold
        clone_qualifiers = max_scales <= self.config.percent_dense * extent
        split_qualifiers = max_scales > self.config.percent_dense * extent

        all_clones = torch.logical_and(clone_qualifiers, grad_qualifiers)
        all_splits = torch.logical_and(split_qualifiers, grad_qualifiers)
        total_clones = torch.sum(all_clones).item()
        total_splits = torch.sum(all_splits).item()

        curr_points = gaussian_model.n_gaussians
        budget = min(self.counts_array[self.densify_iter_num], total_clones + total_splits + curr_points)
        clone_budget = ((budget - curr_points) * total_clones) // (total_clones + total_splits)
        split_budget = ((budget - curr_points) * total_splits) // (total_clones + total_splits)

        self._densify_and_clone(
            scores,
            clone_budget,
            all_clones,
            gaussian_model,
            optimizers,
        )
        self._densify_and_split(
            scores,
            split_budget,
            all_splits,
            gaussian_model,
            optimizers,
        )

    def _densify_and_clone(self, scores, budget, filter, gaussian_model, optimizers):
        scores = scores * filter.float()
        n_init_points = gaussian_model.n_gaussians
        selected_pts_mask = torch.zeros((n_init_points,), dtype=torch.bool, device=scores.device)

        sampled_indices = torch.multinomial(scores, budget, replacement=False)
        selected_pts_mask[sampled_indices] = True

        # Copy selected Gaussians
        new_properties = {}
        for key, value in gaussian_model.properties.items():
            new_properties[key] = value[selected_pts_mask]

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

    def _densify_and_split(self, scores, budget, filter, gaussian_model, optimizers, N: int = 2):
        scores = scores * filter.float()
        n_init_points = gaussian_model.n_gaussians

        padded_importance = torch.zeros((n_init_points,), dtype=scores.dtype, device=scores.device)
        padded_importance[:scores.shape[0]] = scores

        selected_pts_mask = torch.zeros_like(padded_importance, dtype=torch.bool, device=scores.device)
        sampled_indices = torch.multinomial(padded_importance, budget, replacement=False)
        selected_pts_mask[sampled_indices] = True

        # Split
        new_properties = self._split_properties(gaussian_model, selected_pts_mask, N)

        # Update optimizers and properties
        self._densification_postfix(new_properties, gaussian_model, optimizers)

        # Prune selected Gaussians, since they are already split
        prune_filter = torch.cat((
            selected_pts_mask,
            torch.zeros(
                N * selected_pts_mask.sum(),
                device=scores.device,
                dtype=torch.bool,
            ),
        ))
        self._prune_points(prune_filter, gaussian_model, optimizers)

    def _opacity_culling(self, scores, max_screen_size, gaussian_model, optimizers):
        if self.densify_iter_num >= self.config.cull_opacity_until:
            return

        prune_mask = (gaussian_model.get_opacities() < self.config.cull_opacity_threshold).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = gaussian_model.get_scales().max(dim=1).values > 0.1 * self.prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)

        to_remove = torch.sum(prune_mask)
        remove_budget = int(0.5 * to_remove)

        if remove_budget == 0:
            return

        n_init_points = gaussian_model.n_gaussians
        padded_importance = torch.zeros((n_init_points,), dtype=torch.float32)
        padded_importance[:scores.shape[0]] = 1 / (1e-6 + scores.squeeze())
        selected_pts_mask = torch.zeros_like(padded_importance, dtype=torch.bool, device=scores.device)

        sampled_indices = torch.multinomial(padded_importance, remove_budget, replacement=False)
        selected_pts_mask[sampled_indices] = True
        final_prune = torch.logical_and(prune_mask, selected_pts_mask)

        self._prune_points(final_prune, gaussian_model, optimizers)

    def update_states(self, outputs):
        viewspace_point_tensor, visibility_filter = outputs["viewspace_points"], outputs["visibility_filter"]
        if self.config.acc_vis:
            visibility_filter = viewspace_point_tensor.has_hit_any_pixels
        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        # update `max_radii2D` is unnecessary, since it will be reset to 0 after cloning and splitting

        # update states
        xys_grad = viewspace_point_tensor.grad
        if self.config.absgrad is True:
            xys_grad = viewspace_point_tensor.absgrad
        self._add_densification_stats(xys_grad, visibility_filter, scale=viewspace_points_grad_scale)


class Taming3DGSUtils:
    @staticmethod
    def get_edges(image):
        image_pil = transforms.ToPILImage()(image)
        image_gray = image_pil.convert('L')
        image_edges = image_gray.filter(ImageFilter.FIND_EDGES)
        image_edges_tensor = transforms.ToTensor()(image_edges)

        return image_edges_tensor

    @staticmethod
    def get_count_array(start_count, multiplier, densify_until_iter, densify_from_iter, densification_interval, mode):
        # Eq. (2) of taming-3dgs
        if mode == "multiplier":
            budget = int(start_count * float(multiplier))
        elif mode == "final_count":
            budget = multiplier

        num_steps = ((densify_until_iter - densify_from_iter) // densification_interval)
        slope_lower_bound = (budget - start_count) / num_steps

        k = 2 * slope_lower_bound
        a = (budget - start_count - k * num_steps) / (num_steps * num_steps)
        b = k
        c = start_count

        values = [int(1 * a * (x**2) + (b * x) + c) for x in range(num_steps)]

        return values

    @staticmethod
    def compute_photometric_loss(a, b, lambda_dssim: float, mask=None):
        assert mask is None  # TODO: mask

        l1 = torch.abs(a - b).mean()
        ssim = fused_ssim(a.unsqueeze(0), b.unsqueeze(0), train=False)

        return (1. - lambda_dssim) * l1 + lambda_dssim * (1. - ssim)

    @staticmethod
    def get_loss_map(reconstructed_image, original_image, config, edges_loss_norm):
        weights = [config.mse_importance, config.edge_importance]

        l1_loss = torch.mean(torch.abs(reconstructed_image - original_image), 0).detach()
        l1_loss_norm = (l1_loss - torch.min(l1_loss)) / (torch.max(l1_loss) - torch.min(l1_loss))

        final_loss = (weights[0] * l1_loss_norm) + \
            (weights[1] * edges_loss_norm)

        return final_loss

    @staticmethod
    def rasterize_to_weights(opacities, projections, isects, pixel_weights, viewpoint_camera):
        preprocessed_camera = GSplatV1.preprocess_camera(viewpoint_camera)
        img_width, img_height = preprocessed_camera[-1]

        radii, means2d, depths, conics, _ = projections
        _, _, flatten_ids, isect_offsets = isects

        accum_weights, reverse_counts, blend_weights, dist_accum = rasterize_to_weights(
            means2d=means2d,
            conics=conics,
            opacities=opacities.unsqueeze(0),
            image_width=img_width,
            image_height=img_height,
            tile_size=16,
            isect_offsets=isect_offsets,
            flatten_ids=flatten_ids,
            pixel_weights=pixel_weights.unsqueeze(0),
        )

        return depths[0], radii[0], accum_weights[0], reverse_counts[0], blend_weights[0], dist_accum[0]

    @staticmethod
    def vanilla_rasterize_to_weights(pc, renderer, pixel_weights, viewpoint_camera):
        rasterization_outputs = renderer(
            viewpoint_camera,
            pc,
            torch.zeros((3,), dtype=torch.float, device=viewpoint_camera.R.device),
            pixel_weights=pixel_weights,
        )

        return rasterization_outputs["gaussian_depths"], rasterization_outputs["gaussian_radii"], rasterization_outputs["accum_weights"], rasterization_outputs["accum_count"], rasterization_outputs["accum_blend"], rasterization_outputs["accum_dist"]

    @staticmethod
    def normalize(config_value, value_tensor):
        multiplier = config_value
        value_tensor[value_tensor.isnan()] = 0

        valid_indices = (value_tensor > 0)
        valid_value = value_tensor[valid_indices].to(torch.float32)

        ret_value = torch.zeros_like(value_tensor, dtype=torch.float32)
        ret_value[valid_indices] = multiplier * (valid_value / torch.median(valid_value))

        return ret_value

    @classmethod
    def compute_gaussian_score(
        cls,
        gaussian_model,
        renderer,
        global_steps: int,
        all_grads: torch.Tensor,  # [N]
        sample_cameras: List,
        edge_losses: List[torch.Tensor],
        bg_color: torch.Tensor,
        score_coeffs: ScoreCoefficients,
        lambda_dssim: float,
    ):
        num_points = gaussian_model.n_gaussians
        gaussian_importance = torch.zeros(
            (num_points,),
            device=bg_color.device,
            dtype=torch.float32,
        )

        # Is MipSplatting?
        if hasattr(gaussian_model, "get_3d_filtered_scales_and_opacities"):
            _, all_scales = gaussian_model.get_3d_filtered_scales_and_opacities()
        else:
            all_scales = gaussian_model.get_scales()
        all_scales = torch.prod(all_scales, dim=1)

        for camera_idx in range(len(sample_cameras)):
            # TODO: mask
            # TODO: uint8 image
            # TODO: move camera to GPU
            camera, image_info, _ = sample_cameras[camera_idx]
            _, gt_image, masked_pixels = image_info

            gt_image = gt_image.to(device=bg_color.device)

            # appearance model has warm up, so invoke `training_forward` is required
            rgb_rasterization_outputs = renderer.training_forward(
                global_steps,
                None,
                camera,
                gaussian_model,
                bg_color,
                render_types=["rgb"],
            )
            render_image = rgb_rasterization_outputs["render"]
            all_opacity = rgb_rasterization_outputs["opacities"]
            visibility_filter = rgb_rasterization_outputs["visibility_filter"]

            photometric_loss = cls.compute_photometric_loss(
                render_image,
                gt_image,
                lambda_dssim,
            )
            pixel_weights = cls.get_loss_map(render_image, gt_image, score_coeffs, edge_losses[camera_idx].to(device=bg_color.device))  # [H, W]

            all_depths, all_radii, loss_accum, reverse_counts, blending_weights, dist_accum = cls.rasterize_to_weights(
                opacities=rgb_rasterization_outputs["opacities"],
                projections=rgb_rasterization_outputs["projections"],
                isects=rgb_rasterization_outputs["isects"],
                pixel_weights=pixel_weights,
                viewpoint_camera=camera,
            )

            # In gsplat, only the visible Gaussians have valid depth values
            all_depths *= visibility_filter

            g_importance = (
                cls.normalize(score_coeffs.grad_importance, all_grads) +
                cls.normalize(score_coeffs.opac_importance, all_opacity) +
                cls.normalize(score_coeffs.dept_importance, all_depths) +
                cls.normalize(score_coeffs.radii_importance, all_radii) +
                cls.normalize(score_coeffs.scale_importance, all_scales)
            )

            p_importance = (
                cls.normalize(score_coeffs.dist_importance, dist_accum) +
                cls.normalize(score_coeffs.loss_importance, loss_accum) +
                cls.normalize(score_coeffs.count_importance, reverse_counts) +
                cls.normalize(score_coeffs.blend_importance, blending_weights)
            )

            agg_importance = score_coeffs.view_importance * photometric_loss * (p_importance + g_importance) * visibility_filter

            gaussian_importance += agg_importance

        return gaussian_importance
