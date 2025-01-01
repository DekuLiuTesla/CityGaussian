from dataclasses import dataclass
import math
from .renderer import RendererConfig, Renderer, RendererOutputInfo, RendererOutputTypes
from diff_accel_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
import torch


@dataclass
class Taming3DGSRenderer(RendererConfig):
    anti_aliased: bool = False

    filter_2d_kernel_size: float = 0.3

    def instantiate(self, *args, **kwargs) -> "Taming3DGSRendererModule":
        if self.anti_aliased:
            assert self.filter_2d_kernel_size == 0.3
        return Taming3DGSRendererModule(self)


class Taming3DGSRendererModule(Renderer):
    def __init__(self, config: Taming3DGSRenderer):
        super().__init__()
        self.config = config

    def forward(
            self,
            viewpoint_camera,
            pc,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
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
            debug=False,
            antialiasing=self.config.anti_aliased,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        scales = pc.get_scaling
        rotations = pc.get_rotation

        dc = None
        shs = None
        colors_precomp = kwargs.get("colors_precomp", None)
        if colors_precomp is None:
            if pc.is_pre_activated:
                # TODO: avoid slicing
                shs = pc.get_shs()
                dc = shs[:, :1, :]
                shs = shs[:, 1:, :]
            else:
                dc, shs = pc.get_shs_dc(), pc.get_shs_rest()

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, inverse_depth = rasterizer(
            means3D=means3D,
            means2D=means2D,
            dc=dc,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "inverse_depth": inverse_depth,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self):
        return {
            "rgb": RendererOutputInfo("render"),
            "inverse_depth": RendererOutputInfo("inverse_depth", RendererOutputTypes.GRAY),
        }
