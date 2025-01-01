"""
Most codes are copied from https://github.com/hbb1/2d-gaussian-splatting/blob/main/train.py
"""

from typing import Dict, Tuple, Union, Callable, Optional, List

import lightning
import torch
import math
from .renderer import RendererOutputTypes, RendererOutputInfo, Renderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class Vanilla2DGSRenderer(Renderer):
    def __init__(
            self,
            depth_ratio: float = 0.,
    ):
        super().__init__()
        self.depth_ratio = depth_ratio

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

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
        cov3D_precomp = None
        scales = pc.get_scaling[..., :2]
        rotations = pc.get_rotation

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        colors_precomp = kwargs.get("colors_precomp", None)
        shs = None
        if colors_precomp is None:
            # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
            shs = pc.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, allmap = rasterizer(
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
        rets = {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

        # additional regularizations
        render_alpha = allmap[1:2]

        # get normal map
        # transform normal from view space to world space
        render_normal = allmap[2:5]
        render_normal = (render_normal.permute(1, 2, 0) @ (viewpoint_camera.world_to_camera[:3, :3].T)).permute(2, 0, 1)

        # get median depth map
        render_depth_median = allmap[5:6]
        render_depth_median = torch.nan_to_num(render_depth_median, 0, 0)

        # get expected depth map
        render_depth_expected = allmap[0:1]
        render_depth_expected = (render_depth_expected / render_alpha)
        render_depth_expected = torch.nan_to_num(render_depth_expected, 0, 0)

        # get depth distortion map
        render_dist = allmap[6:7]

        # psedo surface attributes
        # surf depth is either median or expected by setting depth_ratio to 1 or 0
        # for bounded scene, use median depth, i.e., depth_ratio = 1;
        # for unbounded scene, use expected depth, i.e., depth_ration = 0, to reduce disk anliasing.
        surf_depth = render_depth_expected * (1 - self.depth_ratio) + (self.depth_ratio) * render_depth_median

        # assume the depth points form the 'surface' and generate psudo surface normal for regularizations.
        surf_normal = self.depth_to_normal(viewpoint_camera, surf_depth)
        surf_normal = surf_normal.permute(2, 0, 1)
        # remember to multiply with accum_alpha since render_normal is unnormalized.
        surf_normal = surf_normal * (render_alpha).detach()

        rets.update({
            'rend_alpha': render_alpha,
            'rend_normal': render_normal,
            'view_normal': -allmap[2:5],
            'rend_dist': render_dist,
            'surf_depth': surf_depth,
            'surf_normal': surf_normal,
        })

        return rets

    @staticmethod
    def depths_to_points(view, depthmap):
        device = view.world_to_camera.device
        c2w = (view.world_to_camera.T).inverse()
        W, H = view.width, view.height
        ndc2pix = torch.tensor([
            [W / 2, 0, 0, W / 2],
            [0, H / 2, 0, H / 2],
            [0, 0, 0, 1]]).float().cuda().T
        projection_matrix = c2w.T @ view.full_projection
        intrins = (projection_matrix @ ndc2pix)[:3, :3].T

        grid_x, grid_y = torch.meshgrid(torch.arange(W, device='cuda').float(), torch.arange(H, device='cuda').float(), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3)
        rays_d = points @ intrins.inverse().T @ c2w[:3, :3].T
        rays_o = c2w[:3, 3]
        points = depthmap.reshape(-1, 1) * rays_d + rays_o
        return points

    @classmethod
    def depth_to_normal(cls, view, depth):
        """
            view: view camera
            depth: depthmap
        """
        points = cls.depths_to_points(view, depth).reshape(*depth.shape[1:], 3)
        output = torch.zeros_like(points)
        dx = torch.cat([points[2:, 1:-1] - points[:-2, 1:-1]], dim=0)
        dy = torch.cat([points[1:-1, 2:] - points[1:-1, :-2]], dim=1)
        normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=-1), dim=-1)
        output[1:-1, 1:-1, :] = normal_map
        return output

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            'render_alpha': RendererOutputInfo("rend_alpha", type=RendererOutputTypes.GRAY),
            'render_normal': RendererOutputInfo("rend_normal", type=RendererOutputTypes.NORMAL_MAP),
            'view_normal': RendererOutputInfo("view_normal", type=RendererOutputTypes.NORMAL_MAP),
            'render_dist': RendererOutputInfo("rend_dist", type=RendererOutputTypes.GRAY),
            'surf_depth': RendererOutputInfo("surf_depth", type=RendererOutputTypes.GRAY),
            'surf_normal': RendererOutputInfo("surf_normal", type=RendererOutputTypes.NORMAL_MAP),
        }
