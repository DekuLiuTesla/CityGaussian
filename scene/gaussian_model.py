#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch_scatter
import traceback
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from pytorch3d.transforms import matrix_to_quaternion
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation, build_symmetric

class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)

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
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

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

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1


class GaussianModelGrad(GaussianModel):
    def __init__(self, sh_degree : int):
        super().__init__(sh_degree)
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz', 'grad_accum', 'denom']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()
        xyz_gradient_accum = self.xyz_gradient_accum.detach().cpu().numpy()
        denom = self.denom.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, xyz_gradient_accum, denom, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        xyz_gradient_accum = np.asarray(plydata.elements[0]["grad_accum"])[..., np.newaxis]
        denom = np.asarray(plydata.elements[0]["denom"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))
        self.xyz_gradient_accum = torch.tensor(xyz_gradient_accum, dtype=torch.float, device="cuda")
        self.denom = torch.tensor(denom, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

class GaussianModelVox(GaussianModel):
    def __init__(self, sh_degree : int, 
                 xyz_range: list, 
                 voxel_size=[0.01, 0.01, 0.01], 
                 mode='gmm',
                 apply_voxelize=False):
        super().__init__(sh_degree)
        self.xyz_range =  torch.tensor(np.asarray(xyz_range)).float().cuda()
        self.voxel_size = torch.tensor(np.asarray(voxel_size)).float().cuda()
        self.mode = mode
        self.apply_voxelize = apply_voxelize
        self.xyz_activation = torch.sigmoid
        self.inverse_xyz_activation = inverse_sigmoid
    
    @property
    def get_xyz(self):
        xyz = self.xyz_activation(self._xyz) * (self.xyz_range[None, 3:] - self.xyz_range[None, :3]) + self.xyz_range[None, :3]
        return xyz
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        mask = torch.logical_and(fused_point_cloud >= self.xyz_range[:3],
                                 fused_point_cloud <= self.xyz_range[3:]).all(dim=1)
        if mask.sum() == 0:
            raise ValueError("Point cloud does not fit in the specified range")
        fused_point_cloud = (fused_point_cloud[mask] - self.xyz_range[:3]) / (self.xyz_range[3:] - self.xyz_range[:3])
        xyz = self.inverse_xyz_activation(fused_point_cloud)
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[mask,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[mask,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales[mask].requires_grad_(True))
        self._rotation = nn.Parameter(rots[mask].requires_grad_(True))
        self._opacity = nn.Parameter(opacities[mask].requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz_range = self.xyz_range.cpu().numpy()
        mask = np.logical_and(xyz >= xyz_range[:3],
                              xyz <= xyz_range[3:]).all(axis=1)
        if mask.sum() == 0:
            raise ValueError("Point cloud does not fit in the specified range")
        xyz = (xyz[mask] - xyz_range[:3]) / (xyz_range[3:] - xyz_range[:3])

        self._xyz = nn.Parameter(self.inverse_xyz_activation(torch.tensor(xyz, dtype=torch.float, device="cuda")).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree

        if self.apply_voxelize:
            points = self.get_xyz
            voxel_index = torch.div(points[:, :3] - self.xyz_range[None, :3], self.voxel_size[None, :], rounding_mode='floor')
            voxel_coords = voxel_index * self.voxel_size[None, :] + self.xyz_range[None, :3] + self.voxel_size[None, :] / 2

            new_coors, unq_inv = self.scatter_gs(voxel_coords, mode=self.mode)
    
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        range_mask = torch.logical_and(new_xyz >= self.xyz_range[:3], new_xyz <= self.xyz_range[3:]).all(dim=1)

        new_xyz = self.inverse_xyz_activation((new_xyz[range_mask] - self.xyz_range[:3]) / (self.xyz_range[3:] - self.xyz_range[:3]))
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))[range_mask]
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)[range_mask]
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)[range_mask]
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)[range_mask]
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)[range_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(range_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)
    
    def scatter_gs(self, coors, mode="gmm", return_inv=True, min_points=0, unq_inv=None, new_coors=None):
        assert self.get_xyz.size(0) == coors.size(0)
        requires_grad = self.get_xyz.requires_grad
        if mode == 'avg':
            mode = 'mean'

        if unq_inv is None and min_points > 0:
            new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        elif unq_inv is None:
            new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])
        
        if mode == "gmm":
            # get mean and cov of GMM within each voxel
            points = self.get_xyz  # [M, 3]
            opacity = self.get_opacity  # [M, 1]
            features = self.get_features  # [M, 16, 3]
            cov3D = build_symmetric(self.get_covariance())  # [M, 3, 3]

            if min_points > 0:
                cnt_per_point = unq_cnt[unq_inv]
                valid_mask = cnt_per_point >= min_points
                points = points[valid_mask]
                opacity = opacity[valid_mask]
                features = features[valid_mask]
                cov3D = cov3D[valid_mask]
                coors = coors[valid_mask]
                new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

            norm_opacity = opacity / torch_scatter.scatter(opacity, unq_inv, dim=0, reduce='sum')[unq_inv]  # [M, 1]
            sub_means = norm_opacity * points  # [M, 3]
            gmm_means = torch_scatter.scatter(sub_means, unq_inv, dim=0, reduce='sum')  # [N, 3]

            deviation = points - gmm_means[unq_inv]  # [M, 3]
            sub_covs = norm_opacity[..., None] * (torch.bmm((deviation).unsqueeze(-1), 
                                                            (deviation).unsqueeze(-2)) + cov3D)  # [M, 3, 3]
            gmm_covs = torch_scatter.scatter(sub_covs, unq_inv, dim=0, reduce='sum')  # [N, 3, 3]
            
            # transform cov to scaling and rotation
            U, S, _ = torch.svd(gmm_covs)  # bound to be symmetric, thus U and V are the same
            q = matrix_to_quaternion(U)  # [N, 4]
            # s = torch.sqrt(S) * torch.norm(q, dim=-1, keepdim=True)
            s = torch.sqrt(S)

            # use probability as feature weights
            inv_gmm_covs = U @ torch.diag_embed(1 / S) @ U.transpose(-1, -2)  # [N, 3, 3]
            weights = opacity * torch.exp(-0.5 * torch.bmm(deviation.unsqueeze(-2), torch.bmm(inv_gmm_covs[unq_inv], deviation.unsqueeze(-1))).squeeze(-1))  # [M, 1]
            weights /= torch_scatter.scatter(weights, unq_inv, dim=0, reduce='sum')[unq_inv]  # [M, 1]

            new_feat = torch_scatter.scatter(features * weights[..., None], unq_inv, dim=0, reduce='sum')  # [N, 16, 3]
            new_opacity = torch_scatter.scatter(opacity * weights, unq_inv, dim=0, reduce='sum')  # [N, 1]
            
            xyz_org = (gmm_means - self.xyz_range[:3]) / (self.xyz_range[3:] - self.xyz_range[:3])
            xyz = self.inverse_xyz_activation(xyz_org)

            self._xyz = nn.Parameter(xyz.requires_grad_(requires_grad))
            self._rotation = nn.Parameter(q.requires_grad_(requires_grad))  # [M, 4]
            self._scaling = nn.Parameter(self.scaling_inverse_activation(s).requires_grad_(requires_grad)) # [M, 3]
            self._opacity = nn.Parameter(self.inverse_opacity_activation(new_opacity).requires_grad_(requires_grad))  # [M, 1]
            self._features_dc = nn.Parameter(new_feat[:, [0]].requires_grad_(requires_grad))
            self._features_rest = nn.Parameter(new_feat[:, 1:].requires_grad_(requires_grad))
            self.max_radii2D = torch.zeros((gmm_means.shape[0]), device="cuda")
        
        else:
            sh_features = self.get_features.reshape(self.get_xyz.shape[0], -1)
            feat = torch.cat([self._xyz, self._scaling, self._rotation, sh_features, 
                            self._opacity, self.max_radii2D[:, None]], dim=-1)
            
            if min_points > 0:
                cnt_per_point = unq_cnt[unq_inv]
                valid_mask = cnt_per_point >= min_points
                feat = feat[valid_mask]
                coors = coors[valid_mask]
                new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
            
            if mode == 'max':
                new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
            elif mode in ('mean', 'sum'):
                new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
            else:
                raise NotImplementedError

            self._xyz = nn.Parameter(new_feat[:, :3].requires_grad_(requires_grad))
            self._scaling = nn.Parameter(new_feat[:, 3:6].requires_grad_(requires_grad))
            self._rotation = nn.Parameter(new_feat[:, 6:10].requires_grad_(requires_grad))
            self._opacity = nn.Parameter(new_feat[:, [-2]].requires_grad_(requires_grad))

            vox_sh_features = new_feat[:, 10:-2].reshape(-1, 16, 3)
            self._features_dc = nn.Parameter(vox_sh_features[:, [0], :].requires_grad_(requires_grad))
            self._features_rest = nn.Parameter(vox_sh_features[:, 1:, :].requires_grad_(requires_grad))
            self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        if not return_inv:
            return new_coors
        else:
            return new_coors, unq_inv


class GaussianModelLoD(GaussianModel):
    def __init__(self, sh_degree : int, 
                 xyz_range: list, 
                 voxel_size=[[0.01, 0.01, 0.01]], 
                 vox_mode='gmm',
                 apply_lod=False,
                 lod_threshold=[0.2]):
        super().__init__(sh_degree)
        self.xyz_range =  torch.tensor(xyz_range).float().cuda()
        self.voxel_size = torch.tensor(voxel_size).float().cuda()
        self.lod_threshold = torch.tensor(lod_threshold+[torch.inf]).float().cuda()
        self.vox_mode = vox_mode
        self.apply_lod = apply_lod
        self.lod_inds = [0]
        self.xyz_activation = torch.sigmoid
        self.inverse_xyz_activation = inverse_sigmoid

        self.num_fine_pts = None
        self.level_lookup = None
    
    @property
    def get_xyz(self):
        xyz = self.xyz_activation(self._xyz) * (self.xyz_range[None, 3:] - self.xyz_range[None, :3]) + self.xyz_range[None, :3]
        return xyz
    
    def create_from_pcd(self, pcd : BasicPointCloud, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        mask = torch.logical_and(fused_point_cloud >= self.xyz_range[:3],
                                 fused_point_cloud <= self.xyz_range[3:]).all(dim=1)
        if mask.sum() == 0:
            raise ValueError("Point cloud does not fit in the specified range")
        fused_point_cloud = (fused_point_cloud[mask] - self.xyz_range[:3]) / (self.xyz_range[3:] - self.xyz_range[:3])
        xyz = self.inverse_xyz_activation(fused_point_cloud)
        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        self._xyz = nn.Parameter(xyz.requires_grad_(True))
        self._features_dc = nn.Parameter(features[mask,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[mask,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales[mask].requires_grad_(True))
        self._rotation = nn.Parameter(rots[mask].requires_grad_(True))
        self._opacity = nn.Parameter(opacities[mask].requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.lod_inds.append(xyz.shape[0])
    
    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        if self.apply_lod:
            level_lookup = self.level_lookup.cpu().numpy()
            attributes = np.concatenate((attributes, level_lookup), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        xyz_range = self.xyz_range.cpu().numpy()
        mask = np.logical_and(xyz >= xyz_range[:3],
                              xyz <= xyz_range[3:]).all(axis=1)
        if mask.sum() == 0:
            raise ValueError("Point cloud does not fit in the specified range")
        xyz = (xyz[mask] - xyz_range[:3]) / (xyz_range[3:] - xyz_range[:3])
        
        
        self._xyz = nn.Parameter(self.inverse_xyz_activation(torch.tensor(xyz, dtype=torch.float, device="cuda")).requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc[mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra[mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        self.active_sh_degree = self.max_sh_degree
        self.lod_inds.append(xyz.shape[0])

        if self.apply_lod:
            try: 
                level_lookup = np.stack((np.asarray(plydata.elements[0]["lookup_0"]),
                                         np.asarray(plydata.elements[0]["lookup_1"])), axis=1)
                self.level_lookup = torch.tensor(level_lookup, dtype=torch.long, device="cuda")
                print("Loaded coarse level GS from file.")
            except:
                points = self.get_xyz
                num_extra_pts = 0
                num_levels = len(self.lod_threshold)-1
                self.num_fine_pts = points.shape[0]
                self.level_lookup = torch.zeros((self._xyz.shape[0], num_levels), dtype=torch.long, device="cuda")

                for level in range(num_levels):
                    voxel_index = torch.div(points[:, :3] - self.xyz_range[None, :3], self.voxel_size[level, :], rounding_mode='floor')
                    voxel_coords = voxel_index * self.voxel_size[level, :] + self.xyz_range[None, :3] + self.voxel_size[level, :] / 2

                    new_coors, unq_inv = self.scatter_gs(voxel_coords, mode=self.vox_mode)
                    num_extra_pts += new_coors.shape[0]
                    self.level_lookup[:self.lod_inds[1], level] = unq_inv + self.lod_inds[level+1]
                
                self.level_lookup = torch.cat([self.level_lookup, torch.zeros((num_extra_pts, 2), dtype=torch.int32, device="cuda")], dim=0)
                print("Generate coarse level GS from scratch.")
        
    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        range_mask = torch.logical_and(new_xyz >= self.xyz_range[:3], new_xyz <= self.xyz_range[3:]).all(dim=1)

        new_xyz = self.inverse_xyz_activation((new_xyz[range_mask] - self.xyz_range[:3]) / (self.xyz_range[3:] - self.xyz_range[:3]))
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))[range_mask]
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)[range_mask]
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)[range_mask]
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)[range_mask]
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)[range_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(range_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def scatter_gs(self, coors, mode="gmm", return_inv=True, min_points=0, unq_inv=None, new_coors=None):
        assert self.lod_inds[1] == coors.size(0)
        if mode == 'avg':
            mode = 'mean'

        if unq_inv is None and min_points > 0:
            new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
        elif unq_inv is None:
            new_coors, unq_inv = torch.unique(coors, return_inverse=True, return_counts=False, dim=0)
        else:
            assert new_coors is not None, 'please pass new_coors for interface consistency, caller: {}'.format(traceback.extract_stack()[-2][2])
        
        if mode == "gmm":
            # get mean and cov of GMM within each voxel
            points = self.get_xyz[self.lod_inds[0]:self.lod_inds[1]]  # [M, 3]
            opacity = self.get_opacity[self.lod_inds[0]:self.lod_inds[1]]  # [M, 1]
            features = self.get_features[self.lod_inds[0]:self.lod_inds[1]]  # [M, 16, 3]
            cov3D = build_symmetric(self.get_covariance()[self.lod_inds[0]:self.lod_inds[1]])  # [M, 3, 3]

            if min_points > 0:
                cnt_per_point = unq_cnt[unq_inv]
                valid_mask = cnt_per_point >= min_points
                points = points[valid_mask]
                opacity = opacity[valid_mask]
                features = features[valid_mask]
                cov3D = cov3D[valid_mask]
                coors = coors[valid_mask]
                new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)

            norm_opacity = opacity / torch_scatter.scatter(opacity, unq_inv, dim=0, reduce='sum')[unq_inv]  # [M, 1]
            sub_means = norm_opacity * points  # [M, 3]
            gmm_means = torch_scatter.scatter(sub_means, unq_inv, dim=0, reduce='sum')  # [N, 3]

            deviation = points - gmm_means[unq_inv]  # [M, 3]
            sub_covs = norm_opacity[..., None] * (torch.bmm((deviation).unsqueeze(-1), 
                                                            (deviation).unsqueeze(-2)) + cov3D)  # [M, 3, 3]
            gmm_covs = torch_scatter.scatter(sub_covs, unq_inv, dim=0, reduce='sum')  # [N, 3, 3]
            
            # transform cov to scaling and rotation
            U, S, _ = torch.svd(gmm_covs)  # bound to be symmetric, thus U and V are the same
            q = matrix_to_quaternion(U)  # [N, 4]
            # s = torch.sqrt(S) * torch.norm(q, dim=-1, keepdim=True)
            s = torch.sqrt(S)

            # use probability as feature weights
            inv_gmm_covs = U @ torch.diag_embed(1 / S) @ U.transpose(-1, -2)  # [N, 3, 3]
            weights = opacity * torch.exp(-0.5 * torch.bmm(deviation.unsqueeze(-2), torch.bmm(inv_gmm_covs[unq_inv], deviation.unsqueeze(-1))).squeeze(-1))  # [M, 1]
            weights /= torch_scatter.scatter(weights, unq_inv, dim=0, reduce='sum')[unq_inv]  # [M, 1]

            new_feat = torch_scatter.scatter(features * weights[..., None], unq_inv, dim=0, reduce='sum')  # [N, 16, 3]
            new_opacity = torch_scatter.scatter(opacity * weights, unq_inv, dim=0, reduce='sum')  # [N, 1]
            
            xyz_org = (gmm_means - self.xyz_range[:3]) / (self.xyz_range[3:] - self.xyz_range[:3])
            xyz = self.inverse_xyz_activation(xyz_org)

            self._xyz = nn.Parameter(torch.cat([self._xyz, xyz.requires_grad_(True)], dim=0))
            self._rotation = nn.Parameter(torch.cat([self._rotation, q.requires_grad_(True)], dim=0))  # [M, 4]
            self._scaling = nn.Parameter(torch.cat([self._scaling, self.scaling_inverse_activation(s).requires_grad_(True)], dim=0)) # [M, 3]
            self._opacity = nn.Parameter(torch.cat([self._opacity, self.inverse_opacity_activation(new_opacity).requires_grad_(True)], dim=0))  # [M, 1]
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, new_feat[:, [0]].requires_grad_(True)], dim=0))
            self._features_rest = nn.Parameter(torch.cat([self._features_rest, new_feat[:, 1:].requires_grad_(True)], dim=0))
            self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")
        
        else:
            sh_features = self.get_features.reshape(self.get_xyz.shape[0], -1)
            feat = torch.cat([self._xyz[self.lod_inds[0]:self.lod_inds[1]], self._scaling[self.lod_inds[0]:self.lod_inds[1]], 
                              self._rotation[self.lod_inds[0]:self.lod_inds[1]], sh_features[self.lod_inds[0]:self.lod_inds[1]], 
                              self._opacity[self.lod_inds[0]:self.lod_inds[1]], self.max_radii2D[self.lod_inds[0]:self.lod_inds[1], None]], dim=-1)
            
            if min_points > 0:
                cnt_per_point = unq_cnt[unq_inv]
                valid_mask = cnt_per_point >= min_points
                feat = feat[valid_mask]
                coors = coors[valid_mask]
                new_coors, unq_inv, unq_cnt = torch.unique(coors, return_inverse=True, return_counts=True, dim=0)
            
            if mode == 'max':
                new_feat, argmax = torch_scatter.scatter_max(feat, unq_inv, dim=0)
            elif mode in ('mean', 'sum'):
                new_feat = torch_scatter.scatter(feat, unq_inv, dim=0, reduce=mode)
            else:
                raise NotImplementedError

            self._xyz = nn.Parameter(torch.cat([self._xyz, new_feat[:, :3].requires_grad_(True)], dim=0))
            self._scaling = nn.Parameter(torch.cat([self._scaling, new_feat[:, 3:6].requires_grad_(True)], dim=0))
            self._rotation = nn.Parameter(torch.cat([self._rotation, new_feat[:, 6:10].requires_grad_(True)], dim=0))
            self._opacity = nn.Parameter(torch.cat([self._opacity, new_feat[:, [-2]].requires_grad_(True)], dim=0))

            vox_sh_features = new_feat[:, 10:-2].reshape(-1, 16, 3)
            self._features_dc = nn.Parameter(torch.cat([self._features_dc, vox_sh_features[:, [0], :].requires_grad_(True)], dim=0))
            self._features_rest = nn.Parameter(torch.cat([self._features_rest, vox_sh_features[:, 1:, :].requires_grad_(True)], dim=0))
            self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        self.lod_inds.append(self._xyz.shape[0])

        if not return_inv:
            return new_coors
        else:
            return new_coors, unq_inv
    
    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        if self.apply_lod:
            for i in range(self.level_lookup.shape[1]):
                l.append('lookup_{}'.format(i))
        return l
