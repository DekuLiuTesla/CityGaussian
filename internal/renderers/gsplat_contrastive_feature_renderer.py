import torch

from gsplat.project_gaussians import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from .renderer import Renderer
from .gsplat_renderer import DEFAULT_BLOCK_SIZE, DEFAULT_ANTI_ALIASED_STATUS
from ..cameras import Camera
from ..models.gaussian import GaussianModel


class GSplatContrastiveFeatureRenderer(Renderer):
    def __init__(self, feature_map_width: int = -1) -> None:
        super().__init__()

        self.block_size = DEFAULT_BLOCK_SIZE
        self.anti_aliased = DEFAULT_ANTI_ALIASED_STATUS
        self.feature_map_width = feature_map_width

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            semantic_features: torch.Tensor = None,
            **kwargs,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())
        fx = viewpoint_camera.fx.item()
        fy = viewpoint_camera.fy.item()
        cx = viewpoint_camera.cx.item()
        cy = viewpoint_camera.cy.item()

        if self.feature_map_width > 0:
            feature_width = self.feature_map_width
            feature_height = int(feature_width * img_height / img_width)

            x_scale = feature_width / img_width
            y_scale = feature_height / img_height

            img_height = feature_height
            img_width = feature_width

            fx = fx * x_scale
            fy = fy * y_scale
            cx = cx * x_scale
            cy = cy * y_scale

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=scaling_modifier,
            quats=pc.get_rotation,
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            # projmat=viewpoint_camera.full_projection.T,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
        )

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        rgb = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            semantic_features,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
            background=bg_color,
            return_alpha=False,
        )  # type: ignore

        return {
            "render": rgb.permute(2, 0, 1),
            "viewspace_points": xys,
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def depth_forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=1.,
            quats=pc.get_rotation,
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            # projmat=viewpoint_camera.full_projection.T,
            fx=viewpoint_camera.fx.item(),
            fy=viewpoint_camera.fy.item(),
            cx=viewpoint_camera.cx.item(),
            cy=viewpoint_camera.cy.item(),
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
        )

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        depth_im = rasterize_gaussians(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            depths.unsqueeze(-1),
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=self.block_size,
            background=torch.zeros((1,), dtype=torch.float, device=xys.device),
            return_alpha=False,
        )  # type: ignore
        depth_im = depth_im.permute(2, 0, 1)

        return depth_im
