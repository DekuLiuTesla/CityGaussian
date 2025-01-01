"""
3D Gaussian Splatting as Markov Chain Monte Carlo
https://ubc-vision.github.io/3dgs-mcmc/

Most codes are copied from https://github.com/ubc-vision/3dgs-mcmc
"""

from typing import Literal, Tuple, Dict, Optional, Union, List
from dataclasses import dataclass
import math
from lightning import LightningModule
import torch
from gsplat.relocation import compute_relocation
from internal.utils.general_utils import inverse_sigmoid
from internal.utils.gaussian_projection import compute_cov_3d

from .density_controller import DensityController, DensityControllerImpl, Utils


@dataclass
class MCMCDensityController(DensityController):
    cap_max: int
    """
    the maximum number of Gaussians
    """

    noise_lr: float = 5e5

    densify_from_iter: int = 500

    densify_until_iter: int = 25_000

    densification_interval: int = 100

    min_opacity: float = 0.005

    N_max: int = 51
    """
    https://github.com/ubc-vision/3dgs-mcmc/blob/main/utils/reloc_utils.py
    """

    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        assert self.cap_max > 0, "cap_max must > 0"
        return MCMCDensityControllerImpl(self)


class MCMCDensityControllerImpl(DensityControllerImpl):
    config: MCMCDensityController

    def setup(self, stage: str, pl_module: LightningModule) -> None:
        super().setup(stage, pl_module)

        # initialize binoms
        N_max = self.config.N_max
        binoms = torch.zeros((N_max, N_max), dtype=torch.float, device=pl_module.device)
        for n in range(N_max):
            for k in range(n + 1):
                binoms[n, k] = math.comb(n, k)

        self.register_buffer("binoms", binoms, persistent=False)

        # initialize opacities and scales
        if stage == "fit":
            self._opacities_and_scales_initialization(pl_module.gaussian_model)

        pl_module.on_train_batch_end_hooks.append(self._add_xyz_noise)

    @staticmethod
    def _opacities_and_scales_initialization(gaussian_model) -> None:
        # looks like it does not affect the final result
        with torch.no_grad():
            gaussian_model.scales.copy_(gaussian_model.scales + math.log(0.1))
            gaussian_model.opacities.copy_(inverse_sigmoid(torch.ones_like(gaussian_model.opacities) * 0.5))

    def after_backward(self, outputs: dict, batch, gaussian_model, optimizers, global_step: int, pl_module: LightningModule) -> None:
        if global_step >= self.config.densify_until_iter:
            return
        if global_step <= self.config.densify_from_iter:
            return
        if global_step % self.config.densification_interval != 0:
            return

        with torch.no_grad():
            dead_mask = (gaussian_model.get_opacities() <= self.config.min_opacity).squeeze(-1)
            # replace based on alive Gaussians
            self.relocate_gs(gaussian_model, optimizers, dead_mask)
            self.add_new_gs(gaussian_model, optimizers)

    @staticmethod
    def op_sigmoid(x, k=100, x0=0.995):
        return 1 / (1 + torch.exp(-k * (x - x0)))

    def _add_xyz_noise(self, outputs: dict, batch, gaussian_model, global_step: int, pl_module: LightningModule) -> None:
        # TODO: really need to add noise till the end of training?

        # prevent adding noise to checkpoint
        if pl_module.is_final_step is True:
            return

        with torch.no_grad():
            cov_3d = compute_cov_3d(
                scales=gaussian_model.get_scales(),
                scale_modifier=1.,
                quaternions=gaussian_model.get_rotations(),
            )

            # TODO: get lr more efficiently
            xyz_lr = -1
            for opt in pl_module.gaussian_optimizers:
                for param_group in opt.param_groups:
                    if param_group["name"] == "means":
                        xyz_lr = param_group["lr"]
                if xyz_lr >= 0:
                    break

            assert xyz_lr >= 0

            noise = torch.randn_like(gaussian_model.means) * (self.op_sigmoid(1 - gaussian_model.get_opacities())) * self.config.noise_lr * xyz_lr
            noise = torch.bmm(cov_3d, noise.unsqueeze(-1)).squeeze(-1)
            gaussian_model.means.add_(noise)

    def compute_relocation(self, opacity_old, scale_old, N) -> Tuple[torch.Tensor, torch.Tensor]:
        # assert torch.all(N <= self.config.N_max)  # whether such a check is necessary?
        return compute_relocation(
            opacity_old,
            scale_old,
            N,
            self.binoms,
        )

    def _get_new_params(self, gaussian_model, idxs, ratio) -> Dict[str, torch.Tensor]:
        # `idxs`: indices of the alive Gaussians
        # `ratio`: sample frequencies of those indices, `ratio[index]=frequency`
        # compute new opacities and scales
        new_opacity, new_scaling = self.compute_relocation(
            opacity_old=gaussian_model.get_opacities()[idxs, 0],  # [N_sample]
            scale_old=gaussian_model.get_scales()[idxs],  # [N_sample]
            N=ratio[idxs, 0] + 1,  # pick the frequencies of those Gaussian to be sampled, [N_sample]
        )
        new_opacity = torch.clamp(new_opacity.unsqueeze(-1), max=1.0 - torch.finfo(torch.float32).eps, min=0.005)
        new_opacity = gaussian_model.opacity_inverse_activation(new_opacity)
        new_scaling = gaussian_model.scale_inverse_activation(new_scaling.reshape(-1, 3))

        new_params = {
            "opacities": new_opacity,
            "scales": new_scaling,
        }

        # get other properties from model, then add to `new_params`
        for attr_name, value in gaussian_model.properties.items():
            if attr_name not in new_params:
                new_params[attr_name] = value[idxs]

        return new_params

    @staticmethod
    def _sample_alives(probs, num, alive_indices=None):
        # `probs` are opacities of alive Gaussians
        # `alive_indices` are the indices of those alive Gaussians
        # `num` is the number of other Gaussians
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)  # pdf
        # Sample `num` Gaussians from alive part. Higher the opacity, higher the sample times
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        # Count the frequency of each value, `ratio[index]=frequency`
        ratio = torch.bincount(sampled_idxs).unsqueeze(-1)
        return sampled_idxs, ratio

    @staticmethod
    def replace_tensors_to_optimizers(gaussian_model, optimizers, inds=None):
        # get current parameters
        properties = gaussian_model.properties

        # replace
        new_parameters = Utils.replace_tensors_to_properties(properties, optimizers=optimizers, selector=inds)

        # update
        gaussian_model.properties = new_parameters

    def relocate_gs(self, gaussian_model, optimizers, dead_mask):
        if dead_mask.sum() == 0:
            return

        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = (gaussian_model.get_opacities()[alive_indices, 0])
        # `reinit_idx` are the sampled alive indices; `ratio` are the values of sample frequency, `ratio[index]=frequency`
        reinit_idx, ratio = self._sample_alives(alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0])

        new_params = self._get_new_params(gaussian_model, reinit_idx, ratio=ratio)
        for attr_name in new_params:
            gaussian_model.get_property(attr_name)[dead_indices] = new_params[attr_name]

        # update the opacities and scales of the sampled Gaussians too
        gaussian_model.opacities[reinit_idx] = gaussian_model.opacities[dead_indices]
        gaussian_model.scales[reinit_idx] = gaussian_model.scales[dead_indices]

        # post-processing: update states of optimizer based on `reinit_idx`, and recreate nn.Parameter for the updated tensors
        # TODO: Why `reinit_idx` only? Should apply to `dead_indices` too?
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=reinit_idx)

    def add_new_gs(self, gaussian_model, optimizers):
        cap_max = self.config.cap_max

        """
        gradually increase the number of live Gaussians by 5% until the maximum desired number of Gaussians is met
        """
        current_num_points = gaussian_model.n_gaussians
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)

        if num_gs <= 0:
            return 0

        probs = gaussian_model.get_opacities().squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        new_params = self._get_new_params(gaussian_model, add_idx, ratio=ratio)

        gaussian_model.opacities[add_idx] = new_params["opacities"]
        gaussian_model.scales[add_idx] = new_params["scales"]

        # densification postfix for new part
        gaussian_model.properties = Utils.cat_tensors_to_properties(new_params, gaussian_model, optimizers)

        # postfix for selected part
        self.replace_tensors_to_optimizers(gaussian_model, optimizers=optimizers, inds=add_idx)

        return num_gs
