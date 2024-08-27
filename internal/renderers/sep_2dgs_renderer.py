from typing import Dict, Tuple, Union, Callable, Optional, List

import lightning
import torch
import math
from .renderer import Renderer
from ..cameras import Camera
from ..models.gaussian_model import GaussianModel

from diff_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer


class Sep2DGSRenderer(Renderer):
    def __init__(
            self,
            depth_ratio: float = 0.,
            lambda_normal: float = 0.05,
            lambda_dist: float = 0.,
            normal_regularization_from_iter: int = 7000,
            dist_regularization_from_iter: int = 3000,
    ):
        super().__init__()
        self.depth_ratio = depth_ratio
        self.lambda_normal = lambda_normal
        self.lambda_dist = lambda_dist
        self.normal_regularization_from_iter = normal_regularization_from_iter
        self.dist_regularization_from_iter = dist_regularization_from_iter

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
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = pc.get_features

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, allmap = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
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

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        with torch.no_grad():
            # key to a quality comparable to hbb1/2d-gaussian-splatting
            if module.hparams["init_from"] is None:
                module.gaussian_model._rotation.copy_(torch.rand_like(module.gaussian_model._rotation))
        return super().training_setup(module)

    @staticmethod
    def depths_to_points(view, depthmap):
        c2w = (view.world_to_camera.T).inverse()
        W, H = view.width.item(), view.height.item()
        fx = W / (2 * math.tan(view.fov_x / 2.))
        fy = H / (2 * math.tan(view.fov_y / 2.))
        intrins = torch.tensor(
            [[fx, 0., W / 2.],
             [0., fy, H / 2.],
             [0., 0., 1.0]]
        ).float().cuda()
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

    def train_metrics(self, pl_module, step: int, batch, outputs):
        metrics, prog_bar = pl_module.vanilla_train_metric_calculator(pl_module, step, batch, outputs)
        camera, _, gt_inverse_depth = batch

        # regularization
        lambda_normal = self.lambda_normal if step > self.normal_regularization_from_iter else 0.0
        lambda_dist = self.lambda_dist if step > self.dist_regularization_from_iter else 0.0

        rend_dist = outputs["rend_dist"]
        rend_normal = outputs['rend_normal']
        surf_normal = outputs['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        normal_loss = lambda_normal * (normal_error).mean()
        dist_loss = lambda_dist * (rend_dist).mean()

        # update metrics
        if step < pl_module.hparams["gaussian"].optimization.densify_until_iter:
            metrics["loss"] = pl_module.lambda_dssim * (1. - metrics["ssim"])
            metrics["extra_loss"] = dist_loss + normal_loss + (1.0 - pl_module.lambda_dssim) * metrics["rgb_diff"]
            prog_bar["extra_loss"] = False
        else:
            metrics["loss"] = metrics["loss"] + dist_loss + normal_loss
        metrics["normal_loss"] = normal_loss
        prog_bar["normal_loss"] = False
        metrics["dist_loss"] = dist_loss
        prog_bar["dist_loss"] = False

        return metrics, prog_bar

    def get_metric_calculators(self) -> Tuple[Union[None, Callable], Union[None, Callable]]:
        return self.train_metrics, None

    def get_available_output_types(self) -> Dict:
        return {
            "rgb": "render",
            'render_alpha': "rend_alpha",
            'render_normal': "rend_normal",
            'view_normal': "view_normal",
            'render_dist': "rend_dist",
            'surf_depth': "surf_depth",
            'surf_normal': "surf_normal",
        }

    def is_type_depth_map(self, t: str) -> bool:
        return t == "surf_depth"

    def is_type_normal_map(self, t: str) -> bool:
        return t == "render_normal" or t == "surf_normal" or t == "view_normal"
