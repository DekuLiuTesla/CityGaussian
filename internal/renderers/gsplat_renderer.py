import torch
from gsplat import project_gaussians
from gsplat.rasterize import rasterize_gaussians
from gsplat.sh import spherical_harmonics
from .renderer import *

DEFAULT_BLOCK_SIZE: int = 16
DEFAULT_ANTI_ALIASED_STATUS: bool = True


class GSPlatRenderer(Renderer):
    _RGB_REQUIRED = 1
    _ALPHA_REQUIRED = 1 << 1
    _ACC_DEPTH_REQUIRED = 1 << 2
    _ACC_DEPTH_INVERTED_REQUIRED = 1 << 3
    _EXP_DEPTH_REQUIRED = 1 << 4
    _EXP_DEPTH_INVERTED_REQUIRED = 1 << 5
    _INVERSE_DEPTH_REQUIRED = 1 << 6
    _HARD_DEPTH_REQUIRED = 1 << 7
    _HARD_INVERSE_DEPTH_REQUIRED = 1 << 8

    RENDER_TYPE_BITS = {
        "rgb": _RGB_REQUIRED,
        "alpha": _ALPHA_REQUIRED | _ACC_DEPTH_REQUIRED,
        "acc_depth": _ACC_DEPTH_REQUIRED,
        "acc_depth_inverted": _ACC_DEPTH_REQUIRED | _ACC_DEPTH_INVERTED_REQUIRED,
        "exp_depth": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED,
        "exp_depth_inverted": _ACC_DEPTH_REQUIRED | _EXP_DEPTH_REQUIRED | _EXP_DEPTH_INVERTED_REQUIRED,
        "inverse_depth": _INVERSE_DEPTH_REQUIRED,
        "hard_depth": _HARD_DEPTH_REQUIRED,
        "hard_inverse_depth": _HARD_INVERSE_DEPTH_REQUIRED,
    }

    def __init__(
            self, 
            block_size: int = DEFAULT_BLOCK_SIZE, 
            anti_aliased: bool = DEFAULT_ANTI_ALIASED_STATUS,
            kernel_size: float = 0.3,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.anti_aliased = anti_aliased
        self.filter_2d_kernel_size = kernel_size

    def parse_render_types(self, render_types: list) -> int:
        if render_types is None:
            return self._RGB_REQUIRED
        else:
            bits = 0
            for i in render_types:
                bits |= self.RENDER_TYPE_BITS[i]
            return bits

    @staticmethod
    def is_type_required(bits: int, type: int) -> bool:
        return bits & type != 0

    def forward(self, viewpoint_camera: Camera, pc: GaussianModel, bg_color: torch.Tensor, scaling_modifier=1.0, render_types: list = None, **kwargs):
        render_type_bits = self.parse_render_types(render_types)

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
            filter_2d_kernel_size=getattr(self, "filter_2d_kernel_size", 0.3),
        )

        opacities = pc.get_opacity
        if self.anti_aliased is True:
            opacities = opacities * comp[:, None]

        def rasterize(input_features: torch.Tensor, background, return_alpha: bool = False):
            return rasterize_gaussians(  # type: ignore
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,  # type: ignore
                input_features,
                opacities,
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=background,
                return_alpha=return_alpha,
            )

        # rgb
        rgb = None
        if self.is_type_required(render_type_bits, self._RGB_REQUIRED):
            viewdirs = pc.get_xyz.detach() - viewpoint_camera.camera_center  # (N, 3)
            rgbs = spherical_harmonics(pc.active_sh_degree, viewdirs, pc.get_features)
            rgbs = torch.clamp(rgbs + 0.5, min=0.0)  # type: ignore

            rgb = rasterize(rgbs, bg_color).permute(2, 0, 1)

        alpha = None
        acc_depth_im = None
        acc_depth_inverted_im = None
        exp_depth_im = None
        exp_depth_inverted_im = None
        if self.is_type_required(render_type_bits, self._ACC_DEPTH_REQUIRED):
            # acc depth
            acc_depth_im, alpha = rasterize(depths.unsqueeze(-1), torch.zeros((1,), device=bg_color.device), True)
            alpha = alpha[..., None]

            # acc depth inverted
            if self.is_type_required(render_type_bits, self._ACC_DEPTH_INVERTED_REQUIRED):
                acc_depth_inverted_im = torch.where(acc_depth_im > 0, 1. / acc_depth_im, acc_depth_im.detach().max())
                acc_depth_inverted_im = acc_depth_inverted_im.permute(2, 0, 1)

            # exp depth
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_REQUIRED):
                exp_depth_im = torch.where(alpha > 0, acc_depth_im / alpha, acc_depth_im.detach().max())

                exp_depth_im = exp_depth_im.permute(2, 0, 1)

            # alpha
            if self.is_type_required(render_type_bits, self._ALPHA_REQUIRED):
                alpha = alpha.permute(2, 0, 1)
            else:
                alpha = None

            # permute acc depth
            acc_depth_im = acc_depth_im.permute(2, 0, 1)

            # exp depth inverted
            if self.is_type_required(render_type_bits, self._EXP_DEPTH_INVERTED_REQUIRED):
                exp_depth_inverted_im = torch.where(exp_depth_im > 0, 1. / exp_depth_im, exp_depth_im.detach().max())

        # inverse depth
        inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths.clamp_min(0.) + 1e-8).unsqueeze(-1)
            inverse_depth_im = rasterize(inverse_depth, torch.zeros((1,), dtype=torch.float, device=bg_color.device)).permute(2, 0, 1)

        # hard depth
        hard_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_DEPTH_REQUIRED):
            hard_depth_im = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                depths.unsqueeze(-1),
                opacities + (1 - opacities.detach()),
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                return_alpha=False,
            ).permute(2, 0, 1)

        # hard inverse depth
        hard_inverse_depth_im = None
        if self.is_type_required(render_type_bits, self._HARD_INVERSE_DEPTH_REQUIRED):
            inverse_depth = 1. / (depths.clamp_min(0.) + 1e-8).unsqueeze(-1)
            hard_inverse_depth_im = rasterize_gaussians(
                xys,
                depths,
                radii,
                conics,
                num_tiles_hit,
                inverse_depth,
                opacities + (1 - opacities.detach()),  # aiming to reduce the opacities of artifacts
                img_height=img_height,
                img_width=img_width,
                block_width=self.block_size,
                background=torch.zeros((1,), dtype=torch.float, device=bg_color.device),
                return_alpha=False,
            ).permute(2, 0, 1)

        return {
            "render": rgb,
            "alpha": alpha,
            "acc_depth": acc_depth_im,
            "acc_depth_inverted": acc_depth_inverted_im,
            "exp_depth": exp_depth_im,
            "exp_depth_inverted": exp_depth_inverted_im,
            "inverse_depth": inverse_depth_im,
            "hard_depth": hard_depth_im,
            "hard_inverse_depth": hard_inverse_depth_im,
            "viewspace_points": xys,
            "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_width, img_height]]).to(xys),
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
            "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_width, img_height]]).to(xys),
            # "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
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
    def rasterize_simplified(project_results, viewpoint_camera, colors, bg_color, opacities, anti_aliased: bool = True):
        xys, depths, radii, conics, comp, num_tiles_hit, cov3d = project_results
        img_height = int(viewpoint_camera.height.item())
        img_width = int(viewpoint_camera.width.item())

        if anti_aliased is True:
            opacities = opacities * comp[:, None]

        return rasterize_gaussians(  # type: ignore
            xys,
            depths,
            radii,
            conics,
            num_tiles_hit,  # type: ignore
            colors,
            opacities,
            img_height=img_height,
            img_width=img_width,
            block_width=DEFAULT_BLOCK_SIZE,
            background=bg_color,
            return_alpha=False,
        ).permute(2, 0, 1)  # type: ignore

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
            "viewspace_points_grad_scale": 0.5 * torch.tensor([[img_width, img_height]]).to(xys),
            # "viewspace_points_grad_scale": 0.5 * max(img_height, img_width),
            "visibility_filter": radii > 0,
            "radii": radii,
        }

    def get_available_outputs(self) -> Dict:
        return {
            "rgb": RendererOutputInfo("render"),
            "alpha": RendererOutputInfo("alpha", type=RendererOutputTypes.GRAY),
            "acc_depth": RendererOutputInfo("acc_depth", type=RendererOutputTypes.GRAY),
            "acc_depth_inverted": RendererOutputInfo("acc_depth_inverted", type=RendererOutputTypes.GRAY),
            "exp_depth": RendererOutputInfo("exp_depth", type=RendererOutputTypes.GRAY),
            "exp_depth_inverted": RendererOutputInfo("exp_depth_inverted", type=RendererOutputTypes.GRAY),
            "inverse_depth": RendererOutputInfo("inverse_depth", type=RendererOutputTypes.GRAY),
            "hard_depth": RendererOutputInfo("hard_depth", type=RendererOutputTypes.GRAY),
            "hard_inverse_depth": RendererOutputInfo("hard_inverse_depth", type=RendererOutputTypes.GRAY),
        }
