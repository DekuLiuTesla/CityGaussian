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

import math
from .renderer import *
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from internal.utils.sh_utils import eval_sh


class VanillaRenderer(Renderer):
    def __init__(self, compute_cov3D_python: bool = False, convert_SHs_python: bool = False):
        super().__init__()

        self.compute_cov3D_python = compute_cov3D_python
        self.convert_SHs_python = convert_SHs_python

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            override_color=None,
            render_types: list = None,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        if render_types is None:
            render_types = ["rgb"]
        assert len(render_types) == 1, "Only single type is allowed currently"

        rendered_image_key = "render"
        if "depth" in render_types:
            rendered_image_key = "depth"
            w2c = viewpoint_camera.world_to_camera  # already transposed
            means3D_in_camera_space = torch.matmul(pc.get_xyz, w2c[:3, :3]) + w2c[3, :3]
            depth = means3D_in_camera_space[:, 2:]
            # bg_color = torch.ones_like(bg_color) * depth.max()
            bg_color = torch.zeros_like(bg_color)
            override_color = depth.repeat(1, 3)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True,
                                              device=bg_color.device) + 0

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
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
        if self.compute_cov3D_python is True:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling
            rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if override_color is None:
            if self.convert_SHs_python is True:
                shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
                dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
                dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
                colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                shs = pc.get_features
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            rendered_image_key: rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    @staticmethod
    def render(
            means3D: torch.Tensor,  # xyz
            opacity: torch.Tensor,
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],
            features: Optional[torch.Tensor],  # shs
            active_sh_degree: int,
            viewpoint_camera,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            colors_precomp: Optional[torch.Tensor] = None,
            cov3D_precomp: Optional[torch.Tensor] = None,
    ):
        if colors_precomp is not None:
            assert features is None
        if cov3D_precomp is not None:
            assert scales is None
            assert rotations is None

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(
            means3D,
            dtype=means3D.dtype,
            requires_grad=True,
            device=means3D.device,
        )

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.fov_x * 0.5)
        tanfovy = math.tan(viewpoint_camera.fov_y * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.height),
            image_width=int(viewpoint_camera.width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_to_camera,
            projmatrix=viewpoint_camera.full_projection,
            sh_degree=active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rasterize_result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=features,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        if len(rasterize_result) == 2:
            rendered_image, radii = rasterize_result
            depth_image = None
        else:
            rendered_image, radii, depth_image = rasterize_result

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "depth": depth_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("depth", RendererOutputTypes.GRAY),
        }
