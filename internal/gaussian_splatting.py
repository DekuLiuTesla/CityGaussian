import os.path
from typing import Tuple, List, Union, Any

import torch.optim
import torchvision
import wandb
from lightning.pytorch.core.module import MODULE_OPTIMIZERS
from torchmetrics.image import PeakSignalNoiseRatio
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, LRSchedulerPLType, STEP_OUTPUT
import lightning.pytorch.loggers

from internal.configs.model import ModelParams
# from internal.configs.appearance import AppearanceModelParams

from internal.models.gaussian_model import GaussianModel
# from internal.models.appearance_model import AppearanceModel
from internal.renderers import Renderer, VanillaRenderer
from internal.utils.ssim import ssim
from internal.utils.lpips import lpips
from jsonargparse import lazy_instance


class GaussianSplatting(LightningModule):
    def __init__(
            self,
            gaussian: ModelParams,
            save_iterations: List[int],
            camera_extent_factor: float = 1.,
            # enable_appearance_model: bool = False,
            background_color: Tuple[float, float, float] = (0., 0., 0.),
            output_path: str = None,
            save_val_output: bool = False,
            max_save_val_output: int = -1,
            renderer: Renderer = lazy_instance(VanillaRenderer),
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # setup models
        self.gaussian_model = GaussianModel(sh_degree=gaussian.sh_degree)
        # self.appearance_model = None if enable_appearance_model is False else AppearanceModel(
        #     n_input_dims=1,
        #     n_grayscale_factors=appearance.n_grayscale_factors,
        #     n_gammas=appearance.n_gammas,
        #     n_neurons=appearance.n_neurons,
        #     n_hidden_layers=appearance.n_hidden_layers,
        #     n_frequencies=appearance.n_frequencies,
        #     grayscale_factors_activation=appearance.grayscale_factors_activation,
        #     gamma_activation=appearance.gamma_activation,
        # )

        self.optimization_hparams = self.hparams["gaussian"].optimization

        self.renderer = renderer

        # metrics
        self.lambda_dssim = gaussian.optimization.lambda_dssim
        self.psnr = PeakSignalNoiseRatio()

        self.background_color = torch.tensor(background_color, dtype=torch.float32)

        self.batch_size = 1
        self.restored_epoch = 0
        self.restored_global_step = 0

    def _l1_loss(self, predict: torch.Tensor, gt: torch.Tensor):
        return torch.abs(predict - gt).mean()

    def _l2_loss(self, predict: torch.Tensor, gt: torch.Tensor):
        return torch.mean((predict - gt) ** 2)

    def setup(self, stage: str):
        if stage == "fit":
            self.gaussian_model.create_from_pcd(
                self.trainer.datamodule.point_cloud,
                deivce=self.device,
            )

        self.renderer.setup(stage, lightning_module=self)

        # use different image log method based on the logger type
        self.log_image = None
        if isinstance(self.logger, lightning.pytorch.loggers.TensorBoardLogger):
            self.log_image = self.tensorboard_log_image
        elif isinstance(self.logger, lightning.pytorch.loggers.WandbLogger):
            self.log_image = self.wandb_log_image

        # set loss function
        self.rgb_diff_loss_fn = self._l1_loss
        if self.hparams["gaussian"].optimization.rgb_diff_loss == "l2":
            print("Use L2 loss")
            self.rgb_diff_loss_fn = self._l2_loss

    def on_load_checkpoint(self, checkpoint) -> None:
        # reinitialize parameters based on the gaussian number in the checkpoint
        self.gaussian_model.initialize_by_gaussian_number(checkpoint["state_dict"]["gaussian_model._xyz"].shape[0])
        # restore some extra parameters
        if "gaussian_model_extra_state_dict" in checkpoint:
            for i in checkpoint["gaussian_model_extra_state_dict"]:
                setattr(self.gaussian_model, i, checkpoint["gaussian_model_extra_state_dict"][i])
            # for previous version
            if "active_sh_degree" not in checkpoint["gaussian_model_extra_state_dict"]:
                self.gaussian_model.active_sh_degree = self.gaussian_model.max_sh_degree

        # get epoch and global_step, which used in the output path of the validation and test images
        self.restored_epoch = checkpoint["epoch"]
        self.restored_global_step = checkpoint["global_step"]

        # call for renderer
        self.renderer.on_load_checkpoint(self, checkpoint)

        super().on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint) -> None:
        # store some extra parameters
        checkpoint["gaussian_model_extra_state_dict"] = {
            "max_radii2D": self.gaussian_model.max_radii2D,
            "xyz_gradient_accum": self.gaussian_model.xyz_gradient_accum,
            "denom": self.gaussian_model.denom,
            "spatial_lr_scale": self.gaussian_model.spatial_lr_scale,
            "active_sh_degree": self.gaussian_model.active_sh_degree,
        }
        super().on_save_checkpoint(checkpoint)

    def tensorboard_log_image(self, tag: str, image_tensor):
        self.logger.experiment.add_image(
            tag,
            image_tensor,
            self.trainer.global_step,
        )

    def wandb_log_image(self, tag: str, image_tensor):
        image_dict = {
            tag: wandb.Image(image_tensor),
        }
        self.logger.experiment.log(
            image_dict,
            step=self.trainer.global_step,
        )

    def forward(self, camera):
        if self.training is True:
            return self.renderer.training_forward(
                self.trainer.global_step,
                self,
                camera,
                self.gaussian_model,
                bg_color=self.background_color.to(camera.R.device),
            )
        return self.renderer(
            camera,
            self.gaussian_model,
            bg_color=self.background_color.to(camera.R.device),
        )

    def forward_with_loss_calculation(self, camera, image_info):
        image_name, gt_image, masked_pixels = image_info

        # forward
        outputs = self(camera)

        image = outputs["render"]

        # calculate loss
        if masked_pixels is not None:
            gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.
        rgb_diff_loss = self.rgb_diff_loss_fn(outputs["render"], gt_image)
        ssim_metric = ssim(outputs["render"], gt_image)
        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1. - ssim_metric)

        return outputs, loss, rgb_diff_loss, ssim_metric

    def optimizers(self, use_pl_optimizer: bool = True):
        optimizers = super().optimizers(use_pl_optimizer=use_pl_optimizer)

        if isinstance(optimizers, list) is False:
            return [optimizers]

        """
        IMPORTANCE: the global_step will be increased on every step() call of all the optimizers,
        issue https://github.com/Lightning-AI/lightning/issues/17958,
        here change _on_before_step and _on_after_step to override this behavior.
        """
        for idx, optimizer in enumerate(optimizers):
            if idx == 0:
                continue
            optimizer._on_before_step = lambda: self.trainer.profiler.start("optimizer_step")
            optimizer._on_after_step = lambda: self.trainer.profiler.stop("optimizer_step")

        return optimizers

    def lr_schedulers(self) -> Union[None, List[LRSchedulerPLType], LRSchedulerPLType]:
        schedulers = super().lr_schedulers()

        if schedulers is None:
            return []

        if isinstance(schedulers, list) is False:
            return [schedulers]

        return schedulers

    def is_final_step(self, step: int = None):
        if step is None:
            step = self.trainer.global_step
        if self.trainer.max_steps > 0 and step >= self.trainer.max_steps:
            return True
        # TODO: make it works when max_epochs set
        return False

    def on_train_start(self) -> None:
        global_step = self.trainer.global_step + 1
        if global_step < self.hparams["gaussian"].optimization.densify_until_iter and self.trainer.world_size > 1:
            print("[WARNING] DDP should only be enabled after finishing densify (after densify_until_iter={} iterations, but {} currently)".format(self.hparams["gaussian"].optimization.densify_until_iter, global_step))
        super().on_train_start()

    def training_step(self, batch, batch_idx):
        camera, image_info = batch
        # image_name, gt_image, masked_pixels = image_info

        global_step = self.trainer.global_step + 1  # must start from 1 to prevent densify at the beginning

        # get optimizers and schedulers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        self.renderer.before_training_step(global_step, self)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if global_step % 1000 == 0:
            self.gaussian_model.oneupSHdegree()

        # forward
        outputs, loss, rgb_diff_loss, ssim_metric = self.forward_with_loss_calculation(camera, image_info)
        image, viewspace_point_tensor, visibility_filter, radii = outputs["render"], outputs["viewspace_points"], \
            outputs["visibility_filter"], outputs["radii"]

        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

        self.log("train/rgb_diff", rgb_diff_loss, on_step=True, on_epoch=False, prog_bar=False, batch_size=self.batch_size)
        self.log("train/ssim", ssim_metric, on_step=True, on_epoch=False, prog_bar=False, batch_size=self.batch_size)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.batch_size)

        # log learning rate and gaussian count every 100 iterations (without plus one step)
        if self.trainer.global_step % 100 == 0:
            metrics_to_log = {
                "train/gaussians_count": self.gaussian_model.get_xyz.shape[0],
            }
            for opt_idx, opt in enumerate(optimizers):
                if opt is None:
                    continue
                for idx, param_group in enumerate(opt.param_groups):
                    param_group_name = param_group["name"] if "name" in param_group else str(idx)
                    metrics_to_log["lr/{}_{}".format(opt_idx, param_group_name)] = param_group["lr"]
            self.logger.log_metrics(
                metrics_to_log,
                step=self.trainer.global_step,
            )

        # backward
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        self.manual_backward(loss)

        # save before densification
        # checkpoint will always be saved after final step, so do not save for final step here
        if global_step in self.hparams["save_iterations"] and self.is_final_step(global_step) is False:
            self.save_gaussian_to_ply()

        # before gradient descend
        with torch.no_grad():
            # Densification
            if global_step < self.hparams["gaussian"].optimization.densify_until_iter:
                # if viewspace_point_tensor.shape[0] != visibility_filter.shape[0]:
                #     # viewspace_point_tensor and radii only contain visible gaussians
                #
                #     original_viewspace_point_tensor = viewspace_point_tensor
                #     original_radii = radii
                #
                #     viewspace_point_tensor = torch.zeros((visibility_filter.shape[0], 2), dtype=original_viewspace_point_tensor.dtype, device=original_viewspace_point_tensor.device)
                #     viewspace_point_tensor.grad = torch.zeros_like(viewspace_point_tensor)
                #     viewspace_point_tensor.grad[visibility_filter] = original_viewspace_point_tensor.grad
                #     radii = torch.zeros((visibility_filter.shape[0],), dtype=original_radii.dtype, device=original_radii.device)
                #     radii[visibility_filter] = original_radii

                gaussians = self.gaussian_model
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, scale=viewspace_points_grad_scale)

                if global_step > self.optimization_hparams.densify_from_iter and global_step % self.optimization_hparams.densification_interval == 0:
                    size_threshold = 20 if global_step > self.optimization_hparams.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        self.hparams["gaussian"].optimization.densify_grad_threshold,
                        0.005,
                        extent=self.cameras_extent,
                        prune_extent=self.prune_extent,
                        max_screen_size=size_threshold,
                    )

                if global_step % self.hparams["gaussian"].optimization.opacity_reset_interval == 0 or \
                        (
                                torch.all(self.background_color == 1.) and global_step == self.hparams[
                            "gaussian"].optimization.densify_from_iter
                        ):
                    gaussians.reset_opacity()

        # optimize
        for optimizer in optimizers:
            optimizer.step()

        # schedule lr
        self.gaussian_model.update_learning_rate(global_step)
        for scheduler in schedulers:
            scheduler.step()

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.renderer.after_training_step(self.trainer.global_step, self)
        super().on_train_batch_end(outputs, batch, batch_idx)

    def validation_step(self, batch, batch_idx, name: str = "val"):
        camera, image_info = batch
        gt_image = image_info[1]

        # forward
        outputs, loss, rgb_diff_loss, ssim_metric = self.forward_with_loss_calculation(camera, image_info)

        self.log(f"{name}/rgb_diff", rgb_diff_loss, on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        self.log(f"{name}/ssim", ssim_metric, on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        self.log(f"{name}/loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        self.log(f"{name}/psnr", self.psnr(outputs["render"], gt_image), on_epoch=True, prog_bar=True,
                 batch_size=self.batch_size)
        self.log(f"{name}/lpips", lpips(outputs["render"], gt_image, net_type='vgg'), on_epoch=True, prog_bar=True,
                 batch_size=self.batch_size)

        # write validation image
        if self.trainer.global_rank == 0 and self.hparams["save_val_output"] is True and (
                self.hparams["max_save_val_output"] < 0 or batch_idx < self.hparams["max_save_val_output"]
        ):
            if self.log_image is not None:
                grid = torchvision.utils.make_grid(torch.concat([outputs["render"], gt_image], dim=-1))
                self.log_image(
                    tag="{}_images/{}".format(name, image_info[0].replace("/", "_")),
                    image_tensor=grid,
                )

            image_output_path = os.path.join(
                self.hparams["output_path"],
                name,
                "epoch={}-step={}".format(
                    max(self.trainer.current_epoch, self.restored_epoch),
                    max(self.trainer.global_step, self.restored_global_step),
                ),
                "{}.png".format(image_info[0].replace("/", "_"))
            )
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            torchvision.utils.save_image(
                torch.concat([outputs["render"], gt_image], dim=-1),
                image_output_path,
            )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx, name="test")

    def configure_optimizers(self):
        self.cameras_extent = self.trainer.datamodule.dataparser_outputs.camera_extent
        self.prune_extent = self.trainer.datamodule.prune_extent
        # gaussian_model.training_setup() must be called here, where parameters have been moved to GPUs
        self.gaussian_model.training_setup(self.hparams["gaussian"].optimization, self.cameras_extent)
        # scale after optimizer being configured, avoid lr scaling
        self.cameras_extent *= self.hparams["camera_extent_factor"]
        self.prune_extent *= self.hparams["camera_extent_factor"]

        # initialize lists that store optimizers and schedulers
        optimizers = [
            self.gaussian_model.optimizer,
        ]
        schedulers = []

        # renderer optimizer and scheduler setup
        renderer_optimizer, renderer_scheduler = self.renderer.training_setup(self)
        if renderer_optimizer is not None:
            optimizers.append(renderer_optimizer)
        if renderer_scheduler is not None:
            schedulers.append(renderer_scheduler)

        return optimizers, schedulers

    def save_gaussian_to_ply(self):
        if self.trainer.global_rank != 0:
            return

        # save ply file
        filename = "point_cloud.ply"
        # if self.trainer.global_rank != 0:
        #     filename = "point_cloud_{}.ply".format(self.trainer.global_rank)
        with torch.no_grad():
            output_dir = os.path.join(self.hparams["output_path"], "point_cloud",
                                      "iteration_{}".format(self.trainer.global_step))
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, filename)
            self.gaussian_model.save_ply(output_path + ".tmp")
            os.rename(output_path + ".tmp", output_path)

        print("Gaussians saved to {}".format(output_path))

        # save checkpoint
        checkpoint_path = os.path.join(
            self.hparams["output_path"],
            "checkpoints",
            "epoch={}-step={}.ckpt".format(self.trainer.current_epoch, self.trainer.global_step),
        )
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        self.trainer.save_checkpoint(checkpoint_path)
        print("checkpoint save to {}".format(checkpoint_path))
