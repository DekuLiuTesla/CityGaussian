from gsplat.sh import spherical_harmonics
from gsplat.rasterize import rasterize_gaussians
from internal.utils.gaussian_projection import project_gaussians
from .renderer import *
from .gsplat_renderer import DEFAULT_BLOCK_SIZE, DEFAULT_ANTI_ALIASED_STATUS


class PythonPreprocessGSplatRenderer(Renderer):
    block_size: int = DEFAULT_BLOCK_SIZE

    anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS

    def __init__(self) -> None:
        super().__init__()

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, **kwargs):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d, mask, rect_min, rect_max = project_gaussians(
            means_3d=pc.get_xyz,
            scales=pc.get_scaling,
            scale_modifier=scaling_modifier,
            quaternions=pc.get_rotation,
            world_to_camera=viewpoint_camera.world_to_camera,
            fx=viewpoint_camera.fx,
            fy=viewpoint_camera.fy,
            cx=viewpoint_camera.cx,
            cy=viewpoint_camera.cy,
            img_height=viewpoint_camera.height,
            img_width=viewpoint_camera.width,
            block_width=self.block_size,
        )

        viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
        viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore
        # rgbs = eval_gaussian_model_sh(viewpoint_camera, pc)

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        rgb = rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            rgbs,
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
            "visibility_filter": mask,
            "radii": radii,
        }
