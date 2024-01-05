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

    if hasattr(pc, "apply_lod") and pc.apply_lod:
        # obtain mask with no grad
        with torch.no_grad():
            assert len(pc.lod_threshold) == (pc.level_lookup.shape[1] + 1)

            pc_xyz = pc.get_xyz
            mask = torch.zeros(pc_xyz.shape[0], dtype=torch.bool, device="cuda")
            
            distance3D = torch.norm(pc_xyz - cam_info["camera_center"], dim=1)
            if pc.lod_threshold[0] > 0:
                mask[:pc.num_fine_pts] |= (distance3D <= pc.lod_threshold[0])[:pc.num_fine_pts]
            for i in range(len(pc.lod_threshold) - 1):
                if pc.lod_threshold[i] >= pc.lod_threshold[i+1] or pc.lod_threshold[i] > pc.max_distance:
                    continue
                mask_fine = (distance3D > pc.lod_threshold[i])[:pc.num_fine_pts] & (distance3D <= pc.lod_threshold[i+1])[:pc.num_fine_pts]
                inds_coarse = pc.level_lookup[:pc.num_fine_pts, i][mask_fine]
                inds_coarse = torch.unique(inds_coarse)
                mask[inds_coarse] = True
            
            assert mask.sum() > 0, "No points were visible in the scene. Please check your LoD parameters."

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = pc.get_xyz[mask]
        means2D = screenspace_points[mask]
        opacity = pc.get_opacity[mask]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)[mask]
        else:
            scales = pc.get_scaling[mask]
            rotations = pc.get_rotation[mask]

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
                dir_pp = (pc.get_xyz[mask] - cam_info["camera_center"].repeat(pc.get_features[mask].shape[0], 1))
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features[mask]
        else:
            colors_precomp = override_color[mask]
        
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

def render_v3(cam_info, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene, with dynamic GS device transport.
    
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

    if hasattr(pc, "apply_lod") and pc.apply_lod:
        # obtain mask with no grad
        with torch.no_grad():
            assert len(pc.lod_threshold) == (pc.level_lookup.shape[1] + 1)

            pc_xyz = pc.get_xyz
            mask = torch.zeros(pc_xyz.shape[0], dtype=torch.bool, device=pc.device)
            cam_center = cam_info["camera_center"].to(pc.device)
            
            distance3D = torch.norm(pc_xyz - cam_center, dim=1)
            if pc.lod_threshold[0] > 0:
                mask[:pc.num_fine_pts] |= (distance3D <= pc.lod_threshold[0])[:pc.num_fine_pts]
            for i in range(len(pc.lod_threshold) - 1):
                if pc.lod_threshold[i] >= pc.lod_threshold[i+1] or pc.lod_threshold[i] > pc.max_distance:
                    continue
                mask_fine = (distance3D > pc.lod_threshold[i])[:pc.num_fine_pts] & (distance3D <= pc.lod_threshold[i+1])[:pc.num_fine_pts]
                inds_coarse = pc.level_lookup[:pc.num_fine_pts, i][mask_fine]
                inds_coarse = torch.unique(inds_coarse)
                mask[inds_coarse] = True
            
            assert mask.sum() > 0, "No points were visible in the scene. Please check your LoD parameters."

        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        means3D = pc.get_xyz[mask].cuda()
        means2D = screenspace_points[mask].cuda()
        opacity = pc.get_opacity[mask].cuda()

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)[mask].cuda()
        else:
            scales = pc.get_scaling[mask].cuda()
            rotations = pc.get_rotation[mask].cuda()

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if pipe.convert_SHs_python:
                shs_view = pc.get_features[mask].transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2).cuda()
                dir_pp = (pc.get_xyz[mask] - cam_info["camera_center"].repeat(pc.get_features[mask].shape[0], 1)).cuda()
                dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features[mask].cuda()
        else:
            colors_precomp = override_color[mask].cuda()
        
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