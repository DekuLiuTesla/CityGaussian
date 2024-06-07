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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.large_utils import in_frustum

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
    rendered_image, radii = rasterizer(
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
    rendered_image, radii = rasterizer(
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
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_lod(viewpoint_cam, lod_list : list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
        
    # sort cells by distance to camera

    in_frustum_mask = in_frustum(viewpoint_cam, lod_list[-1].cell_corners, lod_list[-1].aabb, lod_list[-1].block_dim)
    in_frustum_indices = in_frustum_mask.nonzero().squeeze(0)
    cam_center = viewpoint_cam.camera_center
    distance3D = torch.norm(lod_list[-1].cell_corners[in_frustum_mask, :, :3] - cam_center[:3], dim=2).min(dim=1).values
    in_frustum_indices = in_frustum_indices[torch.sort(distance3D)[1]]
    distance3D = torch.sort(distance3D)[0]
    
    # # used for BlockedGaussianV3
    out_list = []
    main_device = lod_list[-1].feats.device
    max_sh_degree = lod_list[-1].max_sh_degree
    feat_end_dim = 3 * (max_sh_degree + 1) ** 2 + 4
    
    for i, lod_gs in enumerate(lod_list):
        if i == len(lod_list) - 1 and len(out_list) == 0:
            out_xyz_i, out_feats_i = lod_gs.xyz, lod_gs.feats
        else:
            out_xyz_i, out_feats_i = lod_gs.get_feats(in_frustum_indices, distance3D)
            # out_xyz_i, out_feats_i = lod_gs.get_feats_ptwise(viewpoint_cam)
        
        if out_xyz_i.shape[0] == 0:
            continue
        
        out_i = torch.cat([out_xyz_i.to(main_device), out_feats_i.to(main_device)], dim=1)
        out_list.append(out_i)

    feats = torch.cat(out_list, dim=0)
    # feats = lod_list[2].feats

    means3D = feats[:, :3].float()
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    means2D = screenspace_points
    opacity = feats[:, 3].float()
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = feats[:, feat_end_dim:].float()
    else:
        scales = feats[:, feat_end_dim:feat_end_dim+3].float()
        rotations = feats[:, (feat_end_dim+3):].float()
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        features = feats[:, 4:feat_end_dim].reshape(-1, (max_sh_degree+1)**2, 3).float()
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_cam.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color  # check if requires masking
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=max_sh_degree,
        campos=viewpoint_cam.camera_center, 
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
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
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_lod_v2(viewpoint_cam, lod_list : list, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
        
    # sort cells by distance to camera

    in_frustum_mask = in_frustum(viewpoint_cam, lod_list[-1].cell_corners, lod_list[-1].aabb, lod_list[-1].block_dim)
    in_frustum_indices = in_frustum_mask.nonzero().squeeze(0)
    cam_center = viewpoint_cam.camera_center
    distance3D = torch.norm(lod_list[-1].cell_corners[in_frustum_mask, :, :2] - cam_center[:2], dim=2).min(dim=1).values

    focal_length = 0.5 * viewpoint_cam.image_width / math.tan(viewpoint_cam.FoVx * 0.5)
    nyquist_scalings = 2 * distance3D / focal_length
    avg_scalings = torch.stack([lod_list[i].avg_scalings for i in range(len(lod_list))], dim=0)[:, in_frustum_mask]
    
    # compare avg_scalings with nyquist_scalings to decide which lod to use
    values, lod_indices = torch.max((avg_scalings > nyquist_scalings.unsqueeze(0)).to(torch.uint8), dim=0)
    lod_indices[values==0] = len(lod_list) - 1
    
    # used for BlockedGaussianV3
    out_list = []
    main_device = lod_list[-1].feats.device
    max_sh_degree = lod_list[-1].max_sh_degree
    feat_end_dim = 3 * (max_sh_degree + 1) ** 2 + 4
    
    for lod_idx, lod_gs in enumerate(lod_list):
        out_i = lod_gs.get_feats(in_frustum_indices[lod_indices==lod_idx])
        if out_i.shape[0] == 0:
            continue
        if out_i.device != main_device:
            out_i = torch.cat([out_i[:, :3].to(main_device), out_i[:, 3:].half().to(main_device)], dim=1)
        out_list.append(out_i)

    feats = torch.cat(out_list, dim=0)
    # feats = lod_list[1].feats

    means3D = feats[:, :3].float()
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    means2D = screenspace_points
    opacity = feats[:, 3].float()
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = feats[:, feat_end_dim:].float()
    else:
        scales = feats[:, feat_end_dim:feat_end_dim+3].float()
        rotations = feats[:, (feat_end_dim+3):].float()
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        features = feats[:, 4:feat_end_dim].reshape(-1, (max_sh_degree+1)**2, 3).float()
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_cam.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color  # check if requires masking
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=max_sh_degree,
        campos=viewpoint_cam.camera_center, 
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
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
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_lod_v3(viewpoint_cam, gs_xyz, gs_feats, gs_ids, block_scalings, cell_corners, aabb, block_dim, max_sh_degree, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    
    # sort cells by distance to camera

    in_frustum_mask = in_frustum(viewpoint_cam, cell_corners, aabb, block_dim)
    in_frustum_indices = in_frustum_mask.nonzero().squeeze(0)
    cam_center = viewpoint_cam.camera_center
    distance3D = torch.norm(cell_corners[in_frustum_mask, :, :2] - cam_center[:2], dim=2).min(dim=1).values

    focal_length = 0.5 * viewpoint_cam.image_width / math.tan(viewpoint_cam.FoVx * 0.5)
    nyquist_scalings = 2 * distance3D / focal_length
    avg_scalings = block_scalings[:, in_frustum_mask]
    
    # compare avg_scalings with nyquist_scalings to decide which lod to use
    values, lod_indices = torch.max((avg_scalings > nyquist_scalings.unsqueeze(0)).to(torch.uint8), dim=0)
    lod_indices[values==0] = block_scalings.shape[0] - 1

    in_frustum_indices = in_frustum_indices.squeeze() + lod_indices * block_dim[0] * block_dim[1] * block_dim[2]
    mask = torch.isin(gs_ids, in_frustum_indices.to(gs_feats.device))
    
    # used for BlockedGaussianV3
    feat_end_dim = 3 * (max_sh_degree + 1) ** 2 + 1

    means3D = gs_xyz[mask, :3].float()
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
    means2D = screenspace_points
    opacity = gs_feats[mask, 4].float()
    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = gs_feats[mask, feat_end_dim:].float()
    else:
        scales = gs_feats[mask, feat_end_dim:feat_end_dim+3].float()
        rotations = gs_feats[mask, (feat_end_dim+3):].float()
        
    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        features = gs_feats[mask, 1:feat_end_dim].reshape(-1, (max_sh_degree+1)**2, 3).float()
        if pipe.convert_SHs_python:
            shs_view = features.transpose(1, 2).view(-1, 3, (max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_cam.camera_center.repeat(features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(max_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = features
    else:
        colors_precomp = override_color  # check if requires masking
    
    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_cam.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_cam.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_cam.image_height),
        image_width=int(viewpoint_cam.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_cam.world_view_transform,
        projmatrix=viewpoint_cam.full_proj_transform,
        sh_degree=max_sh_degree,
        campos=viewpoint_cam.camera_center, 
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    rendered_image, radii = rasterizer(
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
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}

def render_viewer(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    if isinstance(pc, GaussianModel):
        return render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
    else:
        return render_lod_v3(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)
