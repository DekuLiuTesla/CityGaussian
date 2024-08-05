from typing import Union

import os

import numpy as np
import torch
from torch import nn

from internal.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from internal.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, strip_symmetric, \
    build_scaling_rotation
from internal.utils.graphics_utils import BasicPointCloud
from internal.utils.gaussian_utils import Gaussian as GaussianParameterUtils


class GaussianModel(nn.Module):
    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    def setup_functions(self):

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = GaussianModel.build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

    def __init__(self, sh_degree: int, extra_feature_dims: int = 0, apply_2dgs: bool = False):
        super().__init__()

        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self.extra_feature_dims = extra_feature_dims
        self.apply_2dgs = apply_2dgs

        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._features_extra = torch.empty(0)

        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0

        self.setup_functions()

    def extra_params_to(self, device, dtype):
        self.max_radii2D = self.max_radii2D.to(device=device, dtype=dtype)
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(device=device, dtype=dtype)
        self.denom = self.denom.to(device=device, dtype=dtype)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    @property
    def get_features_extra(self):
        return self._features_extra

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, deivce):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(deivce)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(deivce))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(deivce)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # the parameter device may be "cpu", so tensor must move to cuda before calling distCUDA2()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).to(deivce)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=deivce)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=deivce))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        features_extra = torch.zeros((fused_point_cloud.shape[0], self.extra_feature_dims), dtype=torch.float, device=deivce)
        self._features_extra = nn.Parameter(features_extra.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)

    def initialize_by_gaussian_number(self, n: int):
        xyz = torch.zeros((n, 3))
        features = torch.zeros((n, 3, (self.max_sh_degree + 1) ** 2))
        features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        feature_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        scaling = torch.zeros((n, 3))
        rotation = torch.zeros((n, 4))
        opacity = torch.zeros((n, 1))

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features_dc.requires_grad_(True))
        self._features_rest = nn.Parameter(feature_rest.requires_grad_(True))
        self._scaling = nn.Parameter(scaling.requires_grad_(True))
        self._rotation = nn.Parameter(rotation.requires_grad_(True))
        self._opacity = nn.Parameter(opacity.requires_grad_(True))

        features_extra = torch.zeros((n, self.extra_feature_dims), dtype=torch.float)
        self._features_extra = nn.Parameter(features_extra.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1))
        self.denom = torch.zeros((self.get_xyz.shape[0], 1))

    def training_setup(self, training_args, scene_extent: float):
        self.spatial_lr_scale = scene_extent
        # override spatial_lr_scale if provided
        if training_args.spatial_lr_scale > 0:
            self.spatial_lr_scale = training_args.spatial_lr_scale

        self.percent_dense = training_args.percent_dense

        # some tensor may still in CPU, move to the same device as the _xyz
        self.extra_params_to(self._xyz.device, self._xyz.dtype)

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            # {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._features_rest], 'lr': training_args.feature_rest_lr_init, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._features_extra], 'lr': training_args.feature_extra_lr_init, "name": "f_extra"},
        ]

        print("spatial_lr_scale={}, learning_rates={}".format(self.spatial_lr_scale, {i["name"]: i["lr"] for i in l}))

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        schedulers = []

        xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                               lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                               lr_delay_mult=training_args.position_lr_delay_mult,
                                               max_steps=training_args.position_lr_max_steps)
        schedulers.append(self.get_lr_updater("xyz", xyz_scheduler_args))

        if training_args.feature_rest_lr_max_steps > 0:
            feature_rest_scheduler_args = get_expon_lr_func(
                lr_init=training_args.feature_rest_lr_init,
                lr_final=training_args.feature_rest_lr_init * training_args.feature_rest_lr_final_factor,
                lr_delay_mult=training_args.feature_rest_lr_final_factor,
                max_steps=training_args.feature_rest_lr_max_steps,
            )
            schedulers.append(self.get_lr_updater("f_rest", feature_rest_scheduler_args))

        if self.extra_feature_dims > 0 and training_args.feature_extra_lr_max_steps > 0:
            feature_extra_scheduler_args = get_expon_lr_func(
                lr_init=training_args.feature_extra_lr_init,
                lr_final=training_args.feature_extra_lr_init * training_args.feature_extra_lr_final_factor,
                lr_delay_mult=training_args.feature_extra_lr_final_factor,
                max_steps=training_args.feature_extra_lr_max_steps,
            )
            schedulers.append(self.get_lr_updater("f_extra", feature_extra_scheduler_args))

        self.schedulers = schedulers

    def get_lr_updater(self, name: str, scheduler):
        for idx, param_group in enumerate(self.optimizer.param_groups):
            if param_group["name"] == name:
                break

        def updater(iteration):
            assert self.optimizer.param_groups[idx]["name"] == name
            self.optimizer.param_groups[idx]["lr"] = scheduler(iteration)

        return updater

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # for param_group in self.optimizer.param_groups:
        #     if param_group["name"] == "xyz":
        #         lr = self.xyz_scheduler_args(iteration)
        #         param_group['lr'] = lr
        #     elif param_group["name"] == "f_rest" and self.feature_rest_scheduler_args is not None:
        #         param_group['lr'] = self.feature_rest_scheduler_args(iteration)
        for i in self.schedulers:
            i(iteration)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._features_extra.shape[1]):
            l.append('f_extra_{}'.format(i))
        return l

    def save_ply(self, path):
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        #
        # xyz = self._xyz.detach().cpu().numpy()
        # normals = np.zeros_like(xyz)
        # f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        # opacities = self._opacity.detach().cpu().numpy()
        # scale = self._scaling.detach().cpu().numpy()
        # rotation = self._rotation.detach().cpu().numpy()
        # f_extra = self._features_extra.detach().cpu().numpy()
        #
        # dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        #
        # elements = np.empty(xyz.shape[0], dtype=dtype_full)
        # attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation, f_extra), axis=1)
        # elements[:] = list(map(tuple, attributes))
        # el = PlyElement.describe(elements, 'vertex')
        # PlyData([el]).write(path)

        GaussianParameterUtils(
            sh_degrees=self.max_sh_degree,
            xyz=self._xyz.detach(),
            opacities=self._opacity.detach(),
            features_dc=self._features_dc.detach(),
            features_rest=self._features_rest.detach(),
            scales=self._scaling.detach(),
            rotations=self._rotation.detach(),
            real_features_extra=self._features_extra.detach(),
        ).to_ply_format().save_to_ply(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, device):

        gaussians = GaussianParameterUtils.load_from_ply(path, sh_degrees=self.max_sh_degree)
        xyz = gaussians.xyz
        features_dc = gaussians.features_dc
        features_rest = gaussians.features_rest
        opacities = gaussians.opacities
        scales = gaussians.scales
        rots = gaussians.rotations
        features_extra = gaussians.real_features_extra

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_rest, dtype=torch.float, device=device).transpose(1, 2).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        self._features_extra = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
    
    def load_ckpt(self, state_dict, device):

        gaussians = GaussianParameterUtils.load_from_state_dict(state_dict=state_dict, sh_degrees=self.max_sh_degree)
        xyz = gaussians.xyz
        features_dc = gaussians.features_dc
        features_rest = gaussians.features_rest
        opacities = gaussians.opacities
        scales = gaussians.scales
        rots = gaussians.rotations
        features_extra = gaussians.real_features_extra

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device=device).requires_grad_(True))
        self._features_dc = nn.Parameter(
            torch.tensor(features_dc, dtype=torch.float, device=device).contiguous().requires_grad_(
                True))
        self._features_rest = nn.Parameter(
            torch.tensor(features_rest, dtype=torch.float, device=device).contiguous().requires_grad_(
                True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device=device).requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        self._features_extra = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_extra = optimizable_tensors["f_extra"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)),
                                                    dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                                                       dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                              new_rotation, new_features_extra):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation,
             "f_extra": new_features_extra, }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._features_extra = optimizable_tensors["f_extra"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values > self.percent_dense * scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N, 1)
        means = torch.zeros((stds.size(0), 3), device=self._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_features_extra = self._features_extra[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_features_extra)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self._xyz.device, dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent, axis_ratio_threshold=0.01):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling,
                                                        dim=1).values <= self.percent_dense * scene_extent)
        if self.apply_2dgs:
            # stop densifying points that are too elongated for stability
            axis_ratio = self.get_scaling.min(dim=1).values / self.get_scaling.max(dim=1).values
            selected_pts_mask = torch.logical_and(selected_pts_mask, axis_ratio > axis_ratio_threshold)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]
        new_features_extra = self._features_extra[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling,
                                   new_rotation, new_features_extra)

    def densify_and_prune(self, max_grad, min_opacity, extent, prune_extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * prune_extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter, scale: Union[float, int, None]):
        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)

        if scale is not None:
            grad_norm = grad_norm * scale

        self.xyz_gradient_accum[update_filter] += grad_norm
        self.denom[update_filter] += 1
