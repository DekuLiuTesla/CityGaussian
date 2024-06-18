from typing import Optional
import torch
from .renderer import Renderer
from .gsplat_renderer import GSPlatRenderer
from gsplat.hit_pixel_count import hit_pixel_count


class GSplatHitPixelCountRenderer(Renderer):
    @staticmethod
    def hit_pixel_count(
            means3D: torch.Tensor,  # xyz
            opacities: torch.Tensor,
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],  # remember to normalize them yourself
            viewpoint_camera,
            scaling_modifier=1.0,
            anti_aliased: bool = True,
            block_size: int = 16,
            extra_projection_kwargs: dict = None,
    ):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = GSPlatRenderer.project(
            means3D=means3D,
            scales=scales,
            rotations=rotations,
            viewpoint_camera=viewpoint_camera,
            scaling_modifier=scaling_modifier,
            block_size=block_size,
            extra_projection_kwargs=extra_projection_kwargs,
        )

        if anti_aliased is True:
            opacities = opacities * comp[:, None]

        count, opacity_score, alpha_score, visibility_score = hit_pixel_count(
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,
            opacities,
            img_height=int(viewpoint_camera.height.item()),
            img_width=int(viewpoint_camera.width.item()),
            block_width=block_size,
        )

        return count, opacity_score, alpha_score, visibility_score
