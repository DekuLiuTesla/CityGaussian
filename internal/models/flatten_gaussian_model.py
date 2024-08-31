from typing import Union

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from internal.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from internal.models.gaussian_model import GaussianModel
from internal.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation, strip_symmetric, \
    build_scaling_rotation
from internal.utils.graphics_utils import BasicPointCloud
from internal.utils.gaussian_utils import Gaussian as GaussianParameterUtils


class FlattenGaussianModel(GaussianModel):
    def __init__(self, sh_degree: int, extra_feature_dims: int = 0, apply_2dgs: bool = False):
        super().__init__(sh_degree, extra_feature_dims, apply_2dgs)
        self.eps_s0 = 1e-8
        self.s0 = torch.empty(0)

    @property
    def get_scaling(self):
        self.s0 = torch.ones_like(self._scaling[:, :1]) * self.eps_s0
        return torch.cat((self.s0, self.scaling_activation(self._scaling[:, [-2, -1]])), dim=1)

    @property
    def get_normal(self):
        R = build_rotation(self.get_rotation)
        gs_normal = R[..., 0]
        gs_normal = F.normalize(gs_normal, dim=1)
        return gs_normal

    def create_from_pcd(self, pcd: BasicPointCloud, device):
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(device)
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().to(device))
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(device)
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        # the parameter device may be "cpu", so tensor must move to cuda before calling distCUDA2()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001).to(device)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 2)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

        features_extra = torch.zeros((fused_point_cloud.shape[0], self.extra_feature_dims), dtype=torch.float, device=device)
        self._features_extra = nn.Parameter(features_extra.requires_grad_(True))

        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self.get_scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._features_extra.shape[1]):
            l.append('f_extra_{}'.format(i))
        return l
    
    def initialize_by_gaussian_number(self, n: int):
        xyz = torch.zeros((n, 3))
        features = torch.zeros((n, 3, (self.max_sh_degree + 1) ** 2))
        features_dc = features[:, :, 0:1].transpose(1, 2).contiguous()
        feature_rest = features[:, :, 1:].transpose(1, 2).contiguous()
        scaling = torch.zeros((n, 2))
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

    def save_ply(self, path):

        inverse_eps_s0 = self.scaling_inverse_activation(torch.tensor(self.eps_s0, device=self._scaling.device))
        _scaling = torch.cat((torch.ones_like(self._scaling[:, :1]) * inverse_eps_s0, self._scaling[:, [-2, -1]]), dim=1)
        scale = _scaling.detach().cpu().numpy()

        GaussianParameterUtils(
            sh_degrees=self.max_sh_degree,
            xyz=self._xyz.detach(),
            opacities=self._opacity.detach(),
            features_dc=self._features_dc.detach(),
            features_rest=self._features_rest.detach(),
            scales=scale.detach(),
            rotations=self._rotation.detach(),
            real_features_extra=self._features_extra.detach(),
        ).to_ply_format().save_to_ply(path)
    
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
        self._scaling = nn.Parameter(torch.tensor(scales[:, 1:], dtype=torch.float, device=device).requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device=device).requires_grad_(True))
        self._features_extra = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device=device).requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

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
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N, 1) / (0.8 * N))[:, 1:]
        new_rotation = self._rotation[selected_pts_mask].repeat(N, 1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N, 1, 1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N, 1, 1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N, 1)
        new_features_extra = self._features_extra[selected_pts_mask].repeat(N, 1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_features_extra)

        prune_filter = torch.cat(
            (selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self._xyz.device, dtype=bool)))
        self.prune_points(prune_filter)

    