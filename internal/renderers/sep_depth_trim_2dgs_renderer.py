from typing import Dict, Tuple, Union, Callable, Optional, List

import lightning
import torch
import math
from .renderer import Renderer
from .vanilla_2dgs_renderer import Vanilla2DGSRenderer
from ..cameras import Camera
from ..models.gaussian import GaussianModel

from diff_trim_surfel_rasterization import GaussianRasterizationSettings, GaussianRasterizer

class SepDepthTrim2DGSRenderer(Vanilla2DGSRenderer):
    def __init__(
            self,
            depth_ratio: float = 0.,
            K: int = 5,
            v_pow: float = 0.1,
            prune_ratio: float = 0.1,
            trim_epoch_interval: int = 5,
            contribution_prune_from_iter : int = 1000,
            contribution_prune_interval: int = 500,
            diable_trimming: bool = False,
    ):
        super().__init__(depth_ratio=depth_ratio)

        # hyper-parameters for trimming
        self.K = K
        self.v_pow = v_pow
        self.prune_ratio = prune_ratio
        self.contribution_prune_from_iter = contribution_prune_from_iter
        self.contribution_prune_interval = contribution_prune_interval
        self.diable_trimming = diable_trimming

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            record_transmittance=False,
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
            record_transmittance=record_transmittance,
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
        output = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )

        if record_transmittance:
            transmittance_sum, num_covered_pixels, radii = output
            transmittance = transmittance_sum / (num_covered_pixels + 1e-6)
            return transmittance
        else:
            rendered_image, radii, allmap = output

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
    
    def before_training_step(
            self,
            step: int,
            module,
    ):
        if step != 1 or self.diable_trimming:
            return
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        device =  module.gaussian_model.get_xyz.device
        top_list = [None, ] * self.K
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                trans = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True
                )
                if top_list[0] is not None:
                    m = trans > top_list[0]
                    if m.any():
                        for i in range(self.K - 1):
                            top_list[self.K - 1 - i][m] = top_list[self.K - 2 - i][m]
                            top_list[0][m] = trans[m]
                else:
                    top_list = [trans.clone() for _ in range(self.K)]

            contribution = torch.stack(top_list, dim=-1).mean(-1)
            # tile = torch.quantile(contribution, self.prune_ratio)
            tile = 0  # only prune invisible points at start
            prune_mask = contribution <= tile
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print("Trimming done.")
        torch.cuda.empty_cache()

    def after_training_step(
            self,
            step: int,
            module,
    ):
        cameras = module.trainer.datamodule.dataparser_outputs.train_set.cameras
        if self.diable_trimming or (step > module.density_controller.config.densify_until_iter) \
           or (step < self.contribution_prune_from_iter) \
           or (step % self.contribution_prune_interval != 0):
           return
        
        device =  module.gaussian_model.get_xyz.device

        top_list = [None, ] * self.K
        with torch.no_grad():
            print("Trimming...")
            for i in range(len(cameras)):
                camera = cameras[i].to_device(device)
                trans = self(
                    camera,
                    module.gaussian_model,
                    bg_color=module._fixed_background_color().to(device),
                    record_transmittance=True
                )
                if top_list[0] is not None:
                    m = trans > top_list[0]
                    if m.any():
                        for i in range(self.K - 1):
                            top_list[self.K - 1 - i][m] = top_list[self.K - 2 - i][m]
                            top_list[0][m] = trans[m]
                else:
                    top_list = [trans.clone() for _ in range(self.K)]

            contribution = torch.stack(top_list, dim=-1).mean(-1)

            tile = torch.quantile(contribution, self.prune_ratio)
            prune_mask = (contribution <= tile)
            module.density_controller._prune_points(prune_mask, module.gaussian_model, module.gaussian_optimizers)
            print("Trimming done.")
        torch.cuda.empty_cache()
