"""
StopThePop: Sorted Gaussian Splatting for View-Consistent Real-time Rendering
https://r4dl.github.io/StopThePop/

pip install dacite git+https://github.com/yzslab/StopThePop-Rasterization.git
"""

import math
from typing import Dict
import dataclasses
from dataclasses import dataclass
from .renderer import RendererConfig, Renderer, RendererOutputInfo
import torch
from diff_stp_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer, ExtendedSettings

from ..cameras import Camera
from ..models.gaussian import GaussianModel


@dataclass
class CullingConfig:
    hierarchical_4x4_culling: bool = True

    rect_bounding: bool = True

    tight_opacity_bounding: bool = True

    tile_based_culling: bool = True


@dataclass
class QueueSizeConfig:
    per_pixel: int = 4

    tile_2x2: int = 8

    tile_4x4: int = 64


@dataclass
class SortConfig:
    queue_sizes: QueueSizeConfig = QueueSizeConfig()
    sort_mode: int = 3
    sort_order: int = 3


@dataclass
class STPRenderer(RendererConfig):
    culling_settings: CullingConfig = CullingConfig()

    load_balancing: bool = True

    proper_ewa_scaling: bool = False

    sort_settings: SortConfig = SortConfig()

    def instantiate(self, *args, **kwargs) -> "STPRendererModule":
        return STPRendererModule(self)


class STPRendererModule(Renderer):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.settings = ExtendedSettings.from_dict(dataclasses.asdict(config))

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            override_color=None,
            **kwargs,
    ):
        if render_types is None:
            render_types = ["rgb"]
        assert len(render_types) == 1, "Only single type is allowed currently"

        rendered_image_key = "render"
        render_depth = False
        if "depth" in render_types:
            rendered_image_key = "depth"
            render_depth = True

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
            inv_viewprojmatrix=torch.linalg.inv(viewpoint_camera.full_projection),
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            settings=self.settings,
            render_depth=render_depth,
            debug=False
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means3D = pc.get_xyz
        means2D = screenspace_points
        opacity = pc.get_opacity

        cov3D_precomp = None
        scales = pc.get_scaling
        rotations = pc.get_rotation

        shs = None
        colors_precomp = None
        if override_color is None:
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

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "depth": RendererOutputInfo("depth"),
        }