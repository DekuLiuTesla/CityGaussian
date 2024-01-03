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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, geometry = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "geometry": geometry,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_v2(cam_info, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Set up rasterization configuration
    tanfovx = math.tan(cam_info["FoVx"] * 0.5)
    tanfovy = math.tan(cam_info["FoVy"] * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(cam_info["image_height"]),
        image_width=int(cam_info["image_width"]),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=cam_info["world_view_transform"],
        projmatrix=cam_info["full_proj_transform"],
        sh_degree=pc.active_sh_degree,
        campos=cam_info["camera_center"],
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    if pc.apply_lod:
        torch.autograd.set_detect_anomaly(True)  # TODO: remove this
        pc_xyz = pc.get_xyz
        pc_opacity = pc.get_opacity
        pc_shs = pc.get_features

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            pc_cov3D_precomp = pc.get_covariance(scaling_modifier)
            tmp_cov3D_precomp = torch.zeros_like(pc_cov3D_precomp[:pc.lod_inds[1]], dtype=pc_cov3D_precomp.dtype, requires_grad=True, device="cuda") + 0
        else:
            pc_scales = pc.get_scaling
            pc_rotations = pc.get_rotation
            tmp_scales = torch.zeros_like(pc_scales[:pc.lod_inds[1]], dtype=pc_scales.dtype, requires_grad=True, device="cuda") + 0
            tmp_rotations = torch.zeros_like(pc_rotations[:pc.lod_inds[1]], dtype=pc_rotations.dtype, requires_grad=True, device="cuda") + 0

        tmp_means3D = torch.zeros_like(pc_xyz[:pc.lod_inds[1]], dtype=pc_xyz.dtype, requires_grad=True, device="cuda") + 0
        tmp_opacity = torch.zeros_like(pc_opacity[:pc.lod_inds[1]], dtype=pc_opacity.dtype, requires_grad=True, device="cuda") + 0
        tmp_shs = torch.zeros_like(pc_shs[:pc.lod_inds[1]], dtype=pc_shs.dtype, requires_grad=True, device="cuda") + 0

        distance3D = torch.norm(pc_xyz[:pc.lod_inds[1]] - cam_info["camera_center"], dim=1)
        assert len(pc.unq_inv_list) == (len(pc.lod_threshold) - 1)

        mask0 = distance3D <= pc.lod_threshold[0]
        tmp_means3D[mask0] = tmp_means3D[mask0] + pc_xyz[:pc.lod_inds[1]][mask0]
        tmp_opacity[mask0] = tmp_opacity[mask0] + pc_opacity[:pc.lod_inds[1]][mask0]
        tmp_shs[mask0] = tmp_shs[mask0] + pc_shs[:pc.lod_inds[1]][mask0]

        if pipe.compute_cov3D_python:
            tmp_cov3D_precomp[mask0] = tmp_cov3D_precomp[mask0] + pc_cov3D_precomp[:pc.lod_inds[1]][mask0]
        else:
            tmp_scales[mask0] = tmp_scales[mask0] + pc_scales[:pc.lod_inds[1]][mask0]
            tmp_rotations[mask0] = tmp_rotations[mask0] + pc_rotations[:pc.lod_inds[1]][mask0]

        for i in range(len(pc.unq_inv_list)):
            mask = (distance3D > pc.lod_threshold[i]) & (distance3D <= pc.lod_threshold[i+1])
            tmp_means3D[mask] = tmp_means3D[mask] + pc_xyz[pc.lod_inds[i+1]:pc.lod_inds[i+2]][pc.unq_inv_list[i]][mask]
            tmp_opacity[mask] = tmp_opacity[mask] + pc_opacity[pc.lod_inds[i+1]:pc.lod_inds[i+2]][pc.unq_inv_list[i]][mask]
            tmp_shs[mask] = tmp_shs[mask] + pc_shs[pc.lod_inds[i+1]:pc.lod_inds[i+2]][pc.unq_inv_list[i]][mask]

            if pipe.compute_cov3D_python:
                tmp_cov3D_precomp[mask] = tmp_cov3D_precomp[mask] + pc_cov3D_precomp[pc.lod_inds[i+1]:pc.lod_inds[i+2]][pc.unq_inv_list[i]][mask]
            else:
                tmp_scales[mask] = tmp_scales[mask] + pc_scales[pc.lod_inds[i+1]:pc.lod_inds[i+2]][pc.unq_inv_list[i]][mask]
                tmp_rotations[mask] = tmp_rotations[mask] + pc_rotations[pc.lod_inds[i+1]:pc.lod_inds[i+2]][pc.unq_inv_list[i]][mask]

        sh_feat = tmp_shs.reshape(tmp_means3D.shape[0], -1)  # [N, 48]
        feat = torch.cat([tmp_means3D, tmp_opacity, sh_feat], dim=1)  # [N, 52]
        if pipe.compute_cov3D_python:
            cat_feat = torch.cat([feat, tmp_cov3D_precomp], dim=1)  # [N, 58]
            _, unq_inv = torch.unique(cat_feat, return_inverse=True, return_counts=False, dim=0)
            new_feat = torch_scatter.scatter(cat_feat, unq_inv, dim=0, reduce='mean')
            cov3D_precomp = new_feat[:, -6:]  # [M, 6]
        else:
            cat_feat = torch.cat([feat, tmp_scales, tmp_rotations], dim=1)  # [N, 59]
            _, unq_inv = torch.unique(cat_feat, return_inverse=True, return_counts=False, dim=0)
            new_feat = torch_scatter.scatter(cat_feat, unq_inv, dim=0, reduce='mean')
            scales = new_feat[:, -7:-4]  # [M, 3]
            rotations = new_feat[:, -4:]  # [M, 4]
        means3D = new_feat[:, :3]  # [M, 3]
        opacity = new_feat[:, 3:4]  # [M, 1]
        shs = new_feat[:, 4:52].reshape(-1, 16, 3)  # [M, 16, 3]
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass
        means2D = screenspace_points

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = shs.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (means3D - cam_info["camera_center"].repeat(shs.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = None
            colors_precomp = override_color
        
    else:
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz - cam_info["camera_center"].repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, geometry = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "geometry": geometry,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}