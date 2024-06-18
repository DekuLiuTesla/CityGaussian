import os.path
import queue
import threading
import traceback
from typing import Tuple, List, Dict, Union, Any
from typing_extensions import Self

import torch.optim
import torchvision
import wandb
import numpy as np
from lightning.pytorch.core.module import MODULE_OPTIMIZERS
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from lightning.pytorch import LightningDataModule, LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler, LRSchedulerPLType, STEP_OUTPUT
import lightning.pytorch.loggers

from internal.viewer.training_viewer import TrainingViewer
from internal.configs.model import ModelParams
# from internal.configs.appearance import AppearanceModelParams
from internal.configs.light_gaussian import LightGaussian

from internal.models.gaussian_model import GaussianModel
# from internal.models.appearance_model import AppearanceModel
from internal.renderers import Renderer, VanillaRenderer
from internal.utils.ssim import ssim
from internal.utils.lpips import lpips
from internal.utils.psnr import color_correct
from jsonargparse import lazy_instance

from internal.utils.sh_utils import eval_sh
from internal.utils.graphics_utils import store_ply

lpips: LearnedPerceptualImagePatchSimilarity


class GaussianSplatting(LightningModule):
    def __init__(
            self,
            gaussian: ModelParams,
            light_gaussian: LightGaussian,
            save_iterations: List[int],
            camera_extent_factor: float = 1.,
            # enable_appearance_model: bool = False,
            background_color: Tuple[float, float, float] = (0., 0., 0.),
            random_background: bool = False,
            output_path: str = None,
            init_from: str = None,
            correct_color: bool = False,
            save_val_output: bool = False,
            max_save_val_output: int = -1,
            renderer: Renderer = lazy_instance(VanillaRenderer),
            absgrad: bool = False,
            save_ply: bool = False,
            web_viewer: bool = False,
    ) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.save_hyperparameters()

        # setup models
        self.gaussian_model = GaussianModel(sh_degree=gaussian.sh_degree, extra_feature_dims=gaussian.extra_feature_dims)
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
        self.light_gaussian_hparams = light_gaussian

        self.renderer = renderer

        # metrics
        self.lambda_dssim = gaussian.optimization.lambda_dssim
        self.psnr = PeakSignalNoiseRatio()
        global lpips  # prevent from storing in state_dict
        lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)

        self.background_color = torch.tensor(background_color, dtype=torch.float32)
        if random_background is True:
            self.get_background_color = self._random_background_color
        else:
            self.get_background_color = self._fixed_background_color

        self.web_viewer: TrainingViewer = None

        self.batch_size = 1
        self.restored_epoch = 0
        self.restored_global_step = 0

        self.max_image_saving_threads = 16
        self.image_queue = queue.Queue(maxsize=self.max_image_saving_threads)
        self.image_saving_threads = []

        # metric calculator
        self.train_metric_calculator = self.vanilla_train_metric_calculator
        self.validation_metric_calculator = self.vanilla_validation_metric_calculator

    def vanilla_train_metric_calculator(self, pl_module, step: int, batch, outputs) -> Tuple[Dict[str, Any], Dict[str, bool]]:
        camera, image_info = batch
        image_name, gt_image, masked_pixels = image_info
        image = outputs["render"]

        # calculate loss
        if masked_pixels is not None:
            gt_image = gt_image.clone()
            gt_image[masked_pixels] = image.detach()[masked_pixels]  # copy masked pixels from prediction to G.T.
        rgb_diff_loss = self.rgb_diff_loss_fn(outputs["render"], gt_image)
        ssim_metric = ssim(outputs["render"], gt_image)
        loss = (1.0 - self.lambda_dssim) * rgb_diff_loss + self.lambda_dssim * (1. - ssim_metric)

        return {
            "loss": loss,
            "rgb_diff": rgb_diff_loss,
            "ssim": ssim_metric,
        }, {
            "loss": True,
            "rgb_diff": False,
            "ssim": False,
        }

    def vanilla_validation_metric_calculator(self, pl_module, batch, outputs) -> Tuple[Dict[str, float], Dict[str, bool]]:
        metrics, prog_bar = self.train_metric_calculator(self, self.trainer.global_step, batch, outputs)

        camera, image_info = batch
        image_name, gt_image, _ = image_info

        metrics["psnr"] = self.psnr(outputs["render"], gt_image)
        prog_bar["psnr"] = True
        metrics["lpips"] = lpips(outputs["render"].clamp(0., 1.).unsqueeze(0), gt_image.unsqueeze(0))
        prog_bar["lpips"] = True

        prog_bar["ssim"] = True

        return metrics, prog_bar

    def log_metrics(
            self,
            metrics: dict,
            prog_bar: dict,
            prefix: str,
            on_step: bool,
            on_epoch: bool,
    ):
        for name in metrics:
            self.log(
                f"{prefix}/{name}",
                metrics[name],
                prog_bar=prog_bar[name],
                on_step=on_step,
                on_epoch=on_epoch,
                batch_size=self.batch_size,
            )

    def _l1_loss(self, predict: torch.Tensor, gt: torch.Tensor):
        return torch.abs(predict - gt).mean()

    def _l2_loss(self, predict: torch.Tensor, gt: torch.Tensor):
        return torch.mean((predict - gt) ** 2)

    def _fixed_background_color(self):
        return self.background_color

    def _random_background_color(self):
        return torch.rand(3)

    def setup(self, stage: str):
        if stage == "fit":
            if self.hparams["init_from"] is None:
                self.gaussian_model.create_from_pcd(
                    self.trainer.datamodule.point_cloud,
                    deivce=self.device,
                )
                self.training = True
            else:
                self.gaussian_model.load_ply(self.hparams["init_from"], device=self.device)
                self.gaussian_model.max_radii2D = torch.zeros((self.gaussian_model.get_xyz.shape[0]), device=self.gaussian_model._xyz.device)
                self.gaussian_model.xyz_gradient_accum = torch.zeros((self.gaussian_model.get_xyz.shape[0], 1), device=self.gaussian_model._xyz.device)
                self.gaussian_model.denom = torch.zeros((self.gaussian_model.get_xyz.shape[0], 1), device=self.gaussian_model._xyz.device)

        self.renderer.setup(stage, lightning_module=self)

        # get metric calculator from renderer if available
        train_metric_calculator, validation_metric_calculator = self.renderer.get_metric_calculators()
        if train_metric_calculator is not None:
            self.train_metric_calculator = train_metric_calculator
        if validation_metric_calculator is not None:
            self.validation_metric_calculator = validation_metric_calculator

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

        if "gaussian_model._features_extra" not in checkpoint["state_dict"]:
            # create an empty `_features_extra`
            checkpoint["state_dict"]["gaussian_model._features_extra"] = torch.zeros_like(self.gaussian_model._features_extra)
            # create an optimizer param group
            new_param_groups = checkpoint["optimizer_states"][0]["param_groups"][0].copy()
            new_param_groups["name"] = "f_extra"
            new_param_groups["lr"] = 0.
            new_param_groups["params"] = [len(checkpoint["optimizer_states"][0]["param_groups"])]
            checkpoint["optimizer_states"][0]["param_groups"].append(new_param_groups)

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
                bg_color=self.get_background_color().to(camera.R.device),
            )
        return self.renderer(
            camera,
            self.gaussian_model,
            bg_color=self._fixed_background_color().to(camera.R.device),
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
        ssim_metric = ssim(outputs["render_org"] if "render_org" in outputs else outputs["render"], gt_image)
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

        if self.hparams["web_viewer"] is True and self.trainer.global_rank == 0:
            if self.trainer.datamodule.hparams["type"] in ["blender", "nsvf", "matrixcity"]:
                up = torch.tensor([0., 0., 1.])
            else:
                c2w = self.trainer.datamodule.dataparser_outputs.train_set.cameras.world_to_camera[:, :3, :3]
                up = c2w[:, :3, 1].mean(dim=0)
                up = -up / torch.linalg.norm(up)
            self.web_viewer = TrainingViewer(
                camera_names=self.trainer.datamodule.dataparser_outputs.train_set.image_names,
                cameras=self.trainer.datamodule.dataparser_outputs.train_set.cameras,
                up_direction=up.cpu().numpy(),
                camera_center=self.trainer.datamodule.dataparser_outputs.train_set.cameras.camera_center.mean(dim=0).cpu().numpy(),
                available_appearance_options=self.trainer.datamodule.dataparser_outputs.appearance_group_ids,
            )
            self.web_viewer.start()

    def on_train_batch_start(self, batch: Any, batch_idx: int):
        if self.web_viewer is not None:
            self.web_viewer.training_step(
                self.gaussian_model,
                self.renderer,
                self._fixed_background_color(),
                self.trainer.global_step,
            )
        return super().on_train_batch_start(batch, batch_idx)

    def training_step(self, batch, batch_idx):
        camera, image_info = batch
        # image_name, gt_image, masked_pixels = image_info

        global_step = self.trainer.global_step + 1  # must start from 1 to prevent densify at the beginning

        # get optimizers and schedulers
        optimizers = self.optimizers()
        schedulers = self.lr_schedulers()

        # zero grad
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)

        # save checkpoint
        # checkpoint will always be saved after final step, so do not save for final step here
        if global_step in self.hparams["save_iterations"] and self.is_final_step(global_step) is False and self.trainer.global_step != self.restored_global_step:
            self.save_gaussians()

        # call renderer hook
        self.renderer.before_training_step(global_step, self)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if global_step % 1000 == 0:
            self.gaussian_model.oneupSHdegree()

        # forward
        outputs = self(camera)
        # metrics
        metrics, prog_bar = self.train_metric_calculator(self, global_step, batch, outputs)
        self.log_metrics(metrics, prog_bar, prefix="train", on_step=True, on_epoch=False)

        image, viewspace_point_tensor, visibility_filter, radii = outputs["render"], outputs["viewspace_points"], \
            outputs["visibility_filter"], outputs["radii"]

        # retrieve viewspace_points_grad_scale if provided
        viewspace_points_grad_scale = outputs.get("viewspace_points_grad_scale", None)

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
        self.manual_backward(metrics["loss"])

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
                if self.hparams["absgrad"] is True:
                    viewspace_point_tensor.grad = viewspace_point_tensor.absgrad
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter, scale=viewspace_points_grad_scale)

                if global_step > self.optimization_hparams.densify_from_iter and global_step % self.optimization_hparams.densification_interval == 0:
                    size_threshold = 20 if global_step > self.optimization_hparams.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        self.hparams["gaussian"].optimization.densify_grad_threshold,
                        self.hparams["gaussian"].optimization.cull_opacity_threshold,
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

    def light_gaussian_prune(self, global_step):
        """
        LightGaussian prune
        """
        if global_step not in self.light_gaussian_hparams.prune_steps:
            return

        from internal.utils.light_gaussian import get_count_and_score
        from internal.utils.light_gaussian import calculate_v_imp_score
        from internal.utils.light_gaussian import get_prune_mask

        # try to detect whether anti aliased enabled
        try:
            anti_aliased = self.renderer.anti_aliased
        except:
            anti_aliased = False

        with torch.no_grad():
            count, score, _, _ = get_count_and_score(
                self.gaussian_model,
                self.trainer.datamodule.dataparser_outputs.train_set.cameras,
                anti_aliased,
            )
            v_list = calculate_v_imp_score(
                self.gaussian_model.get_scaling,
                score,
                self.light_gaussian_hparams.v_pow,
            )

            # TODO: `self.light_gaussian_hparams.prune_steps` should be sorted
            prune_step_index = self.light_gaussian_hparams.prune_steps.index(global_step)
            prune_percent = self.light_gaussian_hparams.prune_percent * (self.light_gaussian_hparams.prune_decay ** prune_step_index)
            prune_mask = get_prune_mask(prune_percent, v_list)

            print(f"number_of_gaussian={self.gaussian_model.get_xyz.shape[0]}, "
                  f"number_to_prune={prune_mask.sum().item()}, "
                  f"prune_percent={prune_percent}, "
                  f"anti_aliased={anti_aliased}")
            self.gaussian_model.prune_points(prune_mask)
            print(f"number_of_gaussian_after_pruning={self.gaussian_model.get_xyz.shape[0]}")

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        # the value of `trainer.global_step` here
        # is the same as the local variable `global_step` in training_step
        global_step = self.trainer.global_step

        self.light_gaussian_prune(global_step)

        self.renderer.after_training_step(self.trainer.global_step, self)
        super().on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> None:
        super().on_validation_batch_start(batch, batch_idx, dataloader_idx)
        if self.web_viewer is not None:
            self.web_viewer.validation_step(
                self.gaussian_model,
                self.renderer,
                self._fixed_background_color(),
                batch_idx,
            )

    def validation_step(self, batch, batch_idx, name: str = "val"):
        camera, image_info = batch
        gt_image = image_info[1]

        # forward
        outputs, loss, rgb_diff_loss, ssim_metric = self.forward_with_loss_calculation(camera, image_info)

        self.log(f"{name}/rgb_diff", rgb_diff_loss, on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        
        self.log(f"{name}/loss", loss, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        if self.hparams["correct_color"]:
            render_np = np.array(outputs["render"][None, ...].cpu(), dtype=np.float64).transpose(0, 2, 3, 1)
            gt_np = np.array(gt_image[None, ...].cpu(), dtype=np.float64).transpose(0, 2, 3, 1)
            outputs["render"] = torch.from_numpy(np.array(color_correct(render_np, gt_np), dtype=np.float32).transpose(0, 3, 1, 2)).contiguous().to(gt_image.device)[0]
        
        self.log(f"{name}/ssim", ssim(outputs["render"], gt_image), on_epoch=True, prog_bar=False, batch_size=self.batch_size)
        self.log(f"{name}/psnr", self.psnr(outputs["render"], gt_image), on_epoch=True, prog_bar=True,
                 batch_size=self.batch_size)
        self.log(f"{name}/lpips", lpips(outputs["render"], gt_image, net_type='vgg'), on_epoch=True, prog_bar=True,
                 batch_size=self.batch_size)

        # write validation image
        if self.trainer.global_rank == 0 and self.hparams["save_val_output"] is True and (
                self.hparams["max_save_val_output"] < 0 or batch_idx < self.hparams["max_save_val_output"]
        ):
            self.image_queue.put({
                "output_image": outputs["render"].cpu(),
                "extra_image": outputs["extra_image"].cpu() if "extra_image" in outputs else None,
                "gt_image": gt_image.cpu(),
                "stage": name,
                "image_name": image_info[0],
                "epoch": max(self.trainer.current_epoch, self.restored_epoch),
                "step": max(self.trainer.global_step, self.restored_global_step),
            })

            # if self.log_image is not None:
            #     grid = torchvision.utils.make_grid(torch.concat([outputs["render"], gt_image], dim=-1))
            #     self.log_image(
            #         tag="{}_images/{}".format(name, image_info[0].replace("/", "_")),
            #         image_tensor=grid,
            #     )
            #
            # image_output_path = os.path.join(
            #     self.hparams["output_path"],
            #     name,
            #     "epoch={}-step={}".format(
            #         max(self.trainer.current_epoch, self.restored_epoch),
            #         max(self.trainer.global_step, self.restored_global_step),
            #     ),
            #     "{}.png".format(image_info[0].replace("/", "_"))
            # )
            # os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            # torchvision.utils.save_image(
            #     torch.concat([outputs["render"], gt_image], dim=-1),
            #     image_output_path,
            # )

    def on_validation_epoch_start(self) -> None:
        super().on_validation_epoch_start()
        if self.hparams["save_val_output"] is True:
            for i in range(self.max_image_saving_threads):
                thread = threading.Thread(target=self.save_images)
                self.image_saving_threads.append(thread)
                thread.start()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        for i in range(len(self.image_saving_threads)):
            self.image_queue.put(None)
        for i in self.image_saving_threads:
            i.join()
        self.image_saving_threads = []

    def on_test_epoch_start(self) -> None:
        super().on_test_epoch_start()
        self.on_validation_epoch_start()

    def on_test_epoch_end(self) -> None:
        super().on_test_epoch_end()
        self.on_validation_epoch_end()

    def save_images(self):
        while True:
            item = self.image_queue.get()
            if item is None:
                break

            try:
                image_list = []
                if item["extra_image"] is not None:
                    image_list.append(item["extra_image"])
                image_list += [item["output_image"], item["gt_image"]]
                image = torch.concat(image_list, dim=-1)

                if self.log_image is not None:
                    grid = torchvision.utils.make_grid(image)
                    self.log_image(
                        tag="{}_images/{}".format(item["stage"], item["image_name"].replace("/", "_")),
                        image_tensor=grid,
                    )

                image_output_path = os.path.join(
                    self.hparams["output_path"],
                    item["stage"],
                    "epoch={}-step={}".format(
                        item["epoch"],
                        item["step"],
                    ),
                    "{}.png".format(item["image_name"].replace("/", "_"))
                )
                os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
                torchvision.utils.save_image(
                    image,
                    image_output_path,
                )
            except:
                traceback.print_exc()

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
            if isinstance(renderer_optimizer, list):
                optimizers += renderer_optimizer
            else:
                optimizers.append(renderer_optimizer)
        if renderer_scheduler is not None:
            if isinstance(renderer_scheduler, list):
                schedulers += renderer_scheduler
            else:
                schedulers.append(renderer_scheduler)

        return optimizers, schedulers

    def save_gaussians(self):
        if self.trainer.global_rank != 0:
            return

        if self.hparams["save_ply"] is True:
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
        with torch.no_grad():
            xyz = self.gaussian_model.get_xyz
            rgb = eval_sh(0, self.gaussian_model.get_features[:, :1, :].transpose(1, 2), None)
            store_ply(os.path.join(
                self.hparams["output_path"],
                "checkpoints",
                "epoch={}-step={}-xyz_rgb.ply".format(self.trainer.current_epoch, self.trainer.global_step),
            ), xyz.cpu().numpy(), ((rgb + 0.5).clamp(min=0., max=1.) * 255).to(torch.int).cpu().numpy())
        print("Checkpoint saved to {}".format(checkpoint_path))

    def to(self, *args: Any, **kwargs: Any) -> Self:
        global lpips
        lpips = lpips.to(*args, **kwargs)
        return super().to(*args, **kwargs)
