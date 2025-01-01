from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union, List
import torch
from tqdm.auto import tqdm
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel


@dataclass
class MipSplattingConfigMixin:
    filter_3d_update_interval: int = 100

    opacity_compensation: bool = True
    """
    https://github.com/autonomousvision/mip-splatting/issues/48
    """


class MipSplattingModelMixin:
    _filter_3d_name: str = "filter_3d"

    def get_extra_property_names(self):
        return super().get_extra_property_names() + [self._filter_3d_name]

    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_pcd(xyz, rgb, property_dict, *args, **kwargs)
        property_dict[self._filter_3d_name] = torch.nn.Parameter(torch.empty(xyz.shape[0], 1), requires_grad=False)

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_number(n, property_dict, *args, **kwargs)
        property_dict[self._filter_3d_name] = torch.nn.Parameter(torch.empty(n, 1), requires_grad=False)

    def training_setup(self, module: "lightning.LightningModule") -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        device = self.get_means().device
        self._train_camera_set = [
            i.to_device(device)
            for i in module.trainer.datamodule.dataparser_outputs.train_set.cameras
        ]
        self.compute_3d_filter()
        return super().training_setup(module)

    def on_train_batch_end(self, step: int, module: "lightning.LightningModule"):
        super().on_train_batch_end(step, module)

        is_update_interval_reach = step % self.config.filter_3d_update_interval == 0
        is_final_interval = module.is_final_step(step + self.config.filter_3d_update_interval)
        if is_update_interval_reach is True and is_final_interval is False:
            self.compute_3d_filter()

    def compute_3d_filter(self):
        self.gaussians[self._filter_3d_name] = MipSplattingUtils.compute_3d_filter(
            tqdm(self._train_camera_set, leave=False, desc="Computing 3D filter"),
            self,
        )

    def get_3d_filtered_scales_and_opacities(self):
        # TODO: return new scales and opacities without changing renderer

        return MipSplattingUtils.apply_3d_filter(
            filter_3d=self.get_3d_filter(),
            opacities=self.get_opacities(),
            scales=self.get_scales(),
            opacity_compensation=self.config.opacity_compensation,
        )

    def get_3d_filter(self):
        return self.gaussians[self._filter_3d_name].detach()


@dataclass
class MipSplatting(MipSplattingConfigMixin, VanillaGaussian):
    filter_3d_update_interval: int = 100

    opacity_compensation: bool = True
    """
    https://github.com/autonomousvision/mip-splatting/issues/48
    """

    def instantiate(self, *args, **kwargs) -> "MipSplattingModel":
        return MipSplattingModel(self)


class MipSplattingModel(MipSplattingModelMixin, VanillaGaussianModel):
    pass


class MipSplattingUtils:
    @staticmethod
    @torch.no_grad()
    def compute_3d_filter(cameras, gaussian_model):
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

            # xyz_to_cam = torch.norm(xyz_cam, dim=1)

            # project to screen space
            valid_depth = xyz_cam[:, 2] > 0.01  # TODO: synchronize with GSPlatRenderer.render()

            x, y, z = xyz_cam[:, 0], xyz_cam[:, 1], xyz_cam[:, 2]
            z = torch.clamp(z, min=0.001)

            # same as multiply with K, convert xyz from camera coordinate to pixel coordinate
            x = x / z * camera.fx + camera.width / 2.0
            y = y / z * camera.fy + camera.height / 2.0

            # in_screen = torch.logical_and(torch.logical_and(x >= 0, x < camera.width), torch.logical_and(y >= 0, y < camera.height))

            # use similar tangent space filtering as in the paper
            in_screen = torch.logical_and(
                torch.logical_and(
                    x >= -0.15 * camera.width,
                    x <= camera.width * 1.15,
                ),
                torch.logical_and(
                    y >= -0.15 * camera.height,
                    y <= 1.15 * camera.height,
                ),
            )

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
        return torch.nn.Parameter(filter_3d[..., None], requires_grad=False)

    @staticmethod
    def apply_3d_filter(
            filter_3d: torch.Tensor,
            opacities: torch.Tensor,
            scales: torch.Tensor,
            opacity_compensation: bool = True,  # https://github.com/autonomousvision/mip-splatting/issues/48
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # apply 3D filter
        scales_square = torch.square(scales)
        scales_after_square = scales_square + torch.square(filter_3d)

        new_opacities = opacities
        if opacity_compensation:
            det1 = scales_square.prod(dim=1)
            det2 = scales_after_square.prod(dim=1)
            coef = torch.sqrt(det1 / det2)
            new_opacities = opacities * coef[..., None]

        return new_opacities, torch.sqrt(scales_after_square)
