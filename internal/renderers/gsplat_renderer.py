#from gsplat import rasterization
from gsplat import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import *

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


class GSPlatRenderer(Renderer):
    def __init__(self, block_size: int = DEFAULT_BLOCK_SIZE, anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS) -> None:
        super().__init__()
        self.block_size = block_size
        self.anti_aliased = anti_aliased

        # self.rasterize_mode = "classic"
        # if anti_aliased is True:
        #     self.rasterize_mode = "antialiased"

    # def forward_v1(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
    #
    #     img_height = int(viewpoint_camera.height.item())
    #     img_width = int(viewpoint_camera.width.item())
    #
    #     if render_types is None:
    #         render_mode = "RGB"
    #         rgb_index = (0, 3)
    #         depth_index = (3, 3)
    #     else:
    #         rgb_index = (0, 0)
    #         depth_index = (0, 0)
    #         render_mode_list = []
    #         bg_color_list = []
    #         if "rgb" in render_types or "alpha" in render_types:
    #             render_mode_list.append("RGB")
    #             bg_color_list.append(bg_color)
    #             rgb_index = (0, 3)
    #         # either acc or exp
    #         if "acc_depth" in render_types:
    #             render_mode_list.append("D")
    #             bg_color_list.append(torch.tensor([0.], device=bg_color.device))
    #             depth_index = (rgb_index[1], rgb_index[1] + 1)
    #         elif "exp_depth" in render_types:
    #             render_mode_list.append("ED")
    #             bg_color_list.append(torch.tensor([0.], device=bg_color.device))
    #             depth_index = (rgb_index[1], rgb_index[1] + 1)
    #         bg_color = torch.concat(bg_color_list, dim=-1)
    #         render_mode = "+".join(render_mode_list)
    #
    #     K = torch.tensor([
    #         [viewpoint_camera.fx, 0, viewpoint_camera.cx],
    #         [0, viewpoint_camera.fy, viewpoint_camera.cy],
    #         [0., 0., 1.]],
    #         device=bg_color.device,
    #     )
    #     renders, alphas, meta = rasterization(
    #         means=pc.get_xyz,
    #         quats=pc.get_rotation,
    #         scales=pc.get_scaling,
    #         opacities=pc.get_opacity.squeeze(-1),
    #         colors=pc.get_features,
    #         sh_degree=pc.active_sh_degree,
    #         viewmats=viewpoint_camera.world_to_camera.T[None, ...],
    #         Ks=K[None, ...],
    #         width=img_width,
    #         height=img_height,
    #         backgrounds=bg_color[None, ...],
    #         render_mode=render_mode,
    #         rasterize_mode=getattr(self, "rasterize_mode", "antialiased"),
    #     )
    #
    #     return {
    #         "render": renders[0][..., rgb_index[0]:rgb_index[1]].permute(2, 0, 1),
    #         "depth": renders[0][..., depth_index[0]:depth_index[1]].permute(2, 0, 1),
    #         "alpha": alphas[0].permute(2, 0, 1),
    #         "viewspace_points": meta["means2d"][0],
    #         "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
    #         "visibility_filter": meta["radii"] > 0,
    #         "radii": meta["radii"],
    #     }

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        if render_types is None:
            render_types = ["rgb"]

        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means3d=pc.get_xyz,
            scales=pc.get_scaling,
            glob_scale=scaling_modifier,
            quats=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),
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

        try:
            xys.retain_grad()
        except:
            pass

        viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
        # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
        rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
        rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        rgb = None
        if "rgb" in render_types:
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
            rgb = rgb.permute(2, 0, 1)

        depth_im = None
        if "depth" in render_types:
            depth_im, alpha = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,  # type: ignore
                depths.unsqueeze(-1).repeat(1, 3),
                opacities,
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=torch.zeros_like(bg_color),
                return_alpha=True,
            )  # type: ignore
            alpha = alpha[..., None]
            depth_im = torch.where(alpha > 0, depth_im / alpha, depth_im.detach().max())
            depth_im = depth_im.permute(2, 0, 1)

        return {
            "render": rgb,
            "depth": depth_im,
            "viewspace_points": xys,
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    @staticmethod
    def render(
            means3D: torch.Tensor,  # xyz
            opacities: torch.Tensor,
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],  # remember to normalize them yourself
            features: Optional[torch.Tensor],  # shs
            active_sh_degree: int,
            viewpoint_camera,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS,
            colors_precomp: Optional[torch.Tensor] = None,
            color_computer: Optional = None,
            block_size: int = DEFAULT_BLOCK_SIZE,
            extra_projection_kwargs: dict = None,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_gaussians(  # type: ignore
            means3d=means3D,
            scales=scales,
            glob_scale=scaling_modifier,
            quats=rotations,
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            # projmat=viewpoint_camera.full_projection.T,
            fx=viewpoint_camera.fx.item(),
            fy=viewpoint_camera.fy.item(),
            cx=viewpoint_camera.cx.item(),
            cy=viewpoint_camera.cy.item(),
            img_height=img_height,
            img_width=img_width,
            block_width=block_size,
            **({} if extra_projection_kwargs is None else extra_projection_kwargs),
        )

        try:
            xys.retain_grad()
        except:
            pass

        if colors_precomp is not None:
            rgbs = colors_precomp
        elif color_computer is not None:
            rgbs = color_computer(locals())
        else:
            viewdirs = means3D.detach() - viewpoint_camera.camera_center  # (N, 3)
            # viewdirs = viewdirs / viewdirs.norm(dim=-1, keepdim=True)
            rgbs = spherical_harmonics(active_sh_degree, viewdirs, features)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

        if anti_aliased is True:
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
            block_width=block_size,
            background=bg_color,
            return_alpha=False,
        )  # type: ignore

        return {
            "render": rgb.permute(2, 0, 1),
            "viewspace_points": xys,
            # "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_height, img_width]]).to(xys),
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    @staticmethod
    def project(
            means3D: torch.Tensor,  # xyz
            scales: Optional[torch.Tensor],
            rotations: Optional[torch.Tensor],  # remember to normalize them yourself
            viewpoint_camera,
            scaling_modifier=1.0,
            block_size: int = DEFAULT_BLOCK_SIZE,
            extra_projection_kwargs: dict = None,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        return project_gaussians(  # type: ignore
            means3d=means3D,
            scales=scales,
            glob_scale=scaling_modifier,
            quats=rotations,
            viewmat=viewpoint_camera.world_to_camera.T[:3, :],
            # projmat=viewpoint_camera.full_projection.T,
            fx=viewpoint_camera.fx.item(),
            fy=viewpoint_camera.fy.item(),
            cx=viewpoint_camera.cx.item(),
            cy=viewpoint_camera.cy.item(),
            img_height=img_height,
            img_width=img_width,
            block_width=block_size,
            **({} if extra_projection_kwargs is None else extra_projection_kwargs),
        )

    @staticmethod
    def rasterize(
            opacities,
            rgbs,
            bg_color,
            project_results: Tuple,
            viewpoint_camera,
            xys_retain_grad: bool = True,
            block_size: int = DEFAULT_BLOCK_SIZE,
            anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS,
    ):
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_results

        if xys_retain_grad is True:
            try:
                xys.retain_grad()
            except:
                pass

        if anti_aliased is True:
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
            block_width=block_size,
            background=bg_color,
            return_alpha=False,
        )  # type: ignore

        return {
            "render": rgb.permute(2, 0, 1),
            "viewspace_points": xys,
            # "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_height, img_width]]).to(xys),
            "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_output_types(self) -> Dict:
        return {
            "rgb": "render",
            "depth": "depth",
        }

    def is_type_depth_map(self, t: str) -> bool:
        return t == "depth"

    # def get_available_output_types(self) -> Dict:
    #     return {
    #         "rgb": "render",
    #         "alpha": "alpha",
    #         "acc_depth": "depth",
    #         "exp_depth": "depth",
    #     }
    #
    # def is_type_depth_map(self, t: str) -> bool:
    #     return t == "acc_depth" or t == "exp_depth"
