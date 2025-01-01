from typing import Tuple, Optional

import lightning
import torch
from .renderer import Renderer, Camera, GaussianModel
from .gsplat_renderer import GSPlatRenderer


class MipSplattingGSplatRenderer(Renderer):
    BLOCK_SIZE: int = 16

    filter_2d_kernel_size: float

    filter_3d_update_interval: int

    filter_3d: torch.nn.Parameter  # [n, 1]

    def __init__(
            self,
            filter_2d_kernel_size: float = 0.1,
            filter_3d_update_interval: int = 100,
    ) -> None:
        super().__init__()

        self.filter_2d_kernel_size = filter_2d_kernel_size
        self.filter_3d_update_interval = filter_3d_update_interval
        self.filter_3d = torch.nn.Parameter(torch.empty(0), requires_grad=False)

    def training_setup(self, module: lightning.LightningModule) -> Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LRScheduler]]:
        self.compute_3d_filter(module.trainer.datamodule.dataparser_outputs.train_set.cameras, module.gaussian_model)
        return None, None

    def after_training_step(self, step: int, module):
        super().before_training_step(step, module)

        # TODO: move 3D filter to model so can be densified and pruned by density controller
        is_after_densification = False
        # optimization_hparams = module.optimization_hparams
        # is_after_densification = step < optimization_hparams.densify_until_iter and \
        #                          step > optimization_hparams.densify_from_iter and \
        #                          step % optimization_hparams.densification_interval == 0
        is_update_interval_reach = step % self.filter_3d_update_interval == 0  # the step start from 1, so this make sure that filter_3d will be calculated at the beginning

        is_final_interval = module.is_final_step(step + self.filter_3d_update_interval)

        if is_after_densification or (is_update_interval_reach is True and is_final_interval is False):
            self.compute_3d_filter(module.trainer.datamodule.dataparser_outputs.train_set.cameras, module.gaussian_model)

    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            **kwargs,
    ):
        opacities, scales = self.get_new_opacities_and_scales(pc.get_opacity, pc.get_scaling)
        return GSPlatRenderer.render(
            means3D=pc.get_xyz,
            opacities=opacities,
            scales=scales,
            rotations=pc.get_rotation / pc.get_rotation.norm(dim=-1, keepdim=True),
            features=pc.get_features,
            active_sh_degree=pc.active_sh_degree,
            viewpoint_camera=viewpoint_camera,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            anti_aliased=True,
            extra_projection_kwargs={
                "filter_2d_kernel_size": self.filter_2d_kernel_size,
            }
        )

    def get_new_opacities_and_scales(self, opacities: torch.Tensor, scales: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # apply 3D filter
        scales_square = torch.square(scales)
        det1 = scales_square.prod(dim=1)

        scales_after_square = scales_square + torch.square(self.filter_3d)
        det2 = scales_after_square.prod(dim=1)
        coef = torch.sqrt(det1 / det2)
        return opacities * coef[..., None], torch.sqrt(scales_after_square)

    @torch.no_grad()
    def compute_3d_filter(self, cameras, gaussian_model):
        print("Computing 3D filter")

        # cameras = [i for i in cameras]
        # random.shuffle(cameras)

        # TODO consider focal length and image width
        xyz = gaussian_model.get_xyz
        distance = torch.ones((xyz.shape[0]), device=xyz.device) * 100000.0
        valid_points = torch.zeros((xyz.shape[0]), device=xyz.device, dtype=torch.bool)

        # we should use the focal length of the highest resolution camera
        focal_length = 0.
        """
        What below section do:
          1. calculate gaussians distance to all camera, and pick the minimum distance for each gaussian
          2. find the maximum focal length
          3. find gaussians that visible by any cameras
        """
        for camera in cameras:

            # transform points to camera space
            R = camera.R.T.to(xyz.device)
            T = camera.T.to(xyz.device)
            xyz_cam = xyz @ R + T[None, :]

            xyz_to_cam = torch.norm(xyz_cam, dim=1)

            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.01  # TODO: synchronize with GSPlatRenderer.render()

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            # same as multiply with K, convert xyz from camera coordinate to pixel coordinate
            x = x / z * camera.fx + camera.width / 2.0
            y = y / z * camera.fy + camera.height / 2.0

            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.width), torch.logical_and(y >= 0, y < camera.height))

            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(torch.logical_and(x >= -0.15 * camera.width, x <= camera.width * 1.15), torch.logical_and(y >= -0.15 * camera.height, y <= 1.15 * camera.height))

            # visible by camera (in the front of camera, inside the image plane after being projected)
            valid = torch.logical_and(valid_depth, in_screen)

            # distance[valid] = torch.min(distance[valid], xyz_to_cam[valid])
            # update the minimum distance by this camera
            distance[valid] = torch.min(distance[valid], z[valid])
            # mark point if visible by any camera
            valid_points = torch.logical_or(valid_points, valid)
            # find maximum focal length
            if focal_length < camera.fx:
                focal_length = camera.fx

        # use maximum distance for invisible gaussians
        distance[~valid_points] = distance[valid_points].max()

        # TODO remove hard coded value
        # TODO box to gaussian transform
        filter_3d = distance / focal_length * (0.2 ** 0.5)
        self.filter_3d = torch.nn.Parameter(filter_3d[..., None], requires_grad=False)

    def on_load_checkpoint(self, module, checkpoint):
        self.filter_3d = torch.nn.Parameter(torch.empty_like(checkpoint["state_dict"]["renderer.filter_3d"]), requires_grad=False)
