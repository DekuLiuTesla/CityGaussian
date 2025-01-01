"""
NeRF in the Wild: Neural Radiance Fields for Unconstrained Photo Collections
    Because optimization only yields appearance embeddings l^a for images in the training set, when evaluating error metrics on test-set images we optimize l^a to match the appearance of the true image using only the left half of each image. Error metrics are evaluated on only the right half of each image, so as to avoid information leakage.
"""

import add_pypath
from typing import Any, Optional, Union, Literal
from dataclasses import dataclass
import os
import torch
import lightning
from internal.dataset import Dataset, CacheDataLoader
from internal.metrics.vanilla_metrics import VanillaMetrics
from internal.utils.gaussian_model_loader import GaussianModelLoader
from lightning.pytorch.loggers import WandbLogger
from utils.auto_hyper_parameter import auto_hyper_parameter
from internal.callbacks import ProgressBar, ValidateOnTrainEnd
import jsonargparse


@dataclass
class Config:
    model: str
    """Trained model dir"""

    data: Optional[str] = None
    """Dataset path"""

    name: Optional[str] = None
    """Experiment name"""

    val_only: bool = False

    optimize_on: Literal["left", "right"] = "left"

    seed: int = 42

    lr_init: float = 1e-2
    lr_final_factor: float = 0.05


class AppearanceEmbeddingOptimizer(lightning.LightningModule):
    def __init__(
            self,
            config: Config,
            ckpt,
    ) -> None:
        super().__init__()
        self.config = config
        self.ckpt = ckpt
        self.metric = None

        # optimize using the left half?
        self.on_left = self.config.optimize_on == "left"

    def setup(self, stage: str) -> None:
        super().setup(stage)
        ckpt = self.ckpt

        model = GaussianModelLoader.initialize_model_from_checkpoint(
            ckpt,
            self.device
        )
        renderer = GaussianModelLoader.initialize_renderer_from_checkpoint(
            ckpt,
            stage,
            self.device,
        )

        model.pre_activate_all_properties()
        model.freeze()

        renderer.model.network.requires_grad_(False)

        self.gaussian_model = model
        self.renderer = renderer
        self.metric = VanillaMetrics().instantiate()
        self.metric.setup(stage, self)

        self.register_buffer("background_color", torch.zeros(3, dtype=torch.float, device=self.device))

    def forward(self, camera):
        return self.renderer(
            camera,
            self.gaussian_model,
            bg_color=self.background_color,
            render_types=["rgb"],
        )

    def training_step(self, batch, batch_idx):
        camera, image_info, extra_data = batch
        image_name, gt_image, mask = image_info

        outputs = self(camera)

        image_width = gt_image.shape[-1]
        half_width = image_width // 2

        width_from = 0
        width_to = half_width
        if not self.on_left:
            # on right
            width_from = half_width
            width_to = image_width

        metrics, _ = self.metric.get_train_metrics(
            self,
            self.gaussian_model,
            self.trainer.global_step,
            (camera, (
                image_name,
                gt_image[..., width_from:width_to],
                mask[..., width_from:width_to] if mask is not None else None,
            ), extra_data),
            {"render": outputs["render"][..., width_from:width_to]}
        )

        self.log("train/loss", metrics["loss"], prog_bar=True, on_step=True, on_epoch=False, batch_size=1)
        self.log("train/ssim", metrics["ssim"], prog_bar=True, on_step=True, on_epoch=False, batch_size=1)
        self.log("optimizer/lr", self.optimizer.param_groups[0]["lr"])

        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        camera, image_info, extra_data = batch
        image_name, gt_image, mask = image_info

        outputs = self(camera)

        image_width = gt_image.shape[-1]
        half_width = image_width // 2

        width_from = half_width
        width_to = image_width
        if not self.on_left:
            # optimize on right, validate on left
            width_from = 0
            width_to = half_width

        metrics, _ = self.metric.get_validate_metrics(
            self,
            self.gaussian_model,
            (camera, (
                image_name,
                gt_image[..., width_from:width_to],
                mask[..., width_from:width_to] if mask is not None else None,
            ), extra_data),
            {"render": outputs["render"][..., width_from:width_to]}
        )

        self.log("val/loss", metrics["loss"], prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/psnr", metrics["psnr"], prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/ssim", metrics["ssim"], prog_bar=True, on_epoch=True, batch_size=1)
        self.log("val/lpips", metrics["lpips"], prog_bar=True, on_epoch=True, batch_size=1)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            lr=self.config.lr_init,
            params=self.renderer.model.embedding.parameters(),
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda iter: self.config.lr_final_factor ** min(iter / self.trainer.max_epochs, 1),
        )

        self.optimizer = optimizer

        return [optimizer], [scheduler]

    def transfer_batch_to_device(self, batch, device: torch.device, dataloader_idx: int):
        assert batch[0].device == device
        return batch

    def _on_device_updated(self):
        if self.metric is not None:
            self.metric.on_parameter_move(device=self.device)

    def to(self, *args: Any, **kwargs: Any):
        super().to(*args, **kwargs)

        self._on_device_updated()

        return self

    def cuda(self, device: Optional[Union[torch.device, int]] = None):
        super().cuda(device)

        self._on_device_updated()

        return self

    def cpu(self):
        super().cpu()

        self._on_device_updated()

        return self


def get_config() -> Config:
    return jsonargparse.CLI(Config, as_positional=True)


def main():
    config = get_config()

    lightning.seed_everything(config.seed)

    # load checkpoint
    ckpt_file = GaussianModelLoader.search_load_file(config.model)
    print(ckpt_file)
    ckpt = torch.load(ckpt_file, map_location="cpu")

    # load dataset
    dataparser_outputs = ckpt["datamodule_hyper_parameters"]["parser"].instantiate(
        ckpt["datamodule_hyper_parameters"]["path"] if config.data is None else config.data,
        os.getcwd(),
        0,
    ).get_outputs()
    n_train_set_images = len(dataparser_outputs.train_set)
    max_steps = auto_hyper_parameter(n_train_set_images)[0]
    max_epochs = max_steps // n_train_set_images * 2

    # instantiate LightningModule
    appearance_embedding_optimizer = AppearanceEmbeddingOptimizer(config, ckpt)
    appearance_embedding_optimizer.setup("fit")

    experiment_name = os.path.basename(ckpt["hyper_parameters"]["output_path"]) if config.name is None else config.name
    experiment_name = "{}-{}".format(experiment_name, config.optimize_on)
    output_dir = os.path.join(os.path.dirname(os.path.dirname(ckpt_file)), "embedding_optimization")
    os.makedirs(output_dir, exist_ok=True)

    # instantiate Trainer
    trainer = lightning.Trainer(
        accelerator="cuda",
        devices=-1 if not config.val_only else 1,
        max_epochs=max_epochs,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=10,
        logger=WandbLogger(
            save_dir=output_dir,
            name=experiment_name,
            project="Embedding",
        ) if not config.val_only else None,
        callbacks=[ProgressBar(), ValidateOnTrainEnd()],
        # profiler="simple",
        enable_checkpointing=False,
        use_distributed_sampler=False,
        log_every_n_steps=min(len(dataparser_outputs.val_set), 50),
    )

    # setup dataloader
    dataloader = CacheDataLoader(
        dataset=Dataset(
            dataparser_outputs.val_set,
            undistort_image=False,
            camera_device=trainer.strategy.root_device,
            image_device=trainer.strategy.root_device,
        ),
        max_cache_num=-1,
        shuffle=True,
        seed=config.seed + trainer.global_rank,
    )

    if config.val_only:
        trainer.validate(appearance_embedding_optimizer, dataloaders=dataloader)
        return
    else:
        # start fitting
        trainer.fit(appearance_embedding_optimizer, train_dataloaders=dataloader, val_dataloaders=dataloader)

    # save
    if trainer.global_rank == 0:
        embedding_state_dict = appearance_embedding_optimizer.renderer.model.embedding.state_dict()
        ckpt = torch.load(ckpt_file, map_location="cpu")
        prefix = "renderer.model.embedding."
        for k in embedding_state_dict:
            key = "{}{}".format(prefix, k)
            assert key in ckpt["state_dict"]
            ckpt["state_dict"][key] = embedding_state_dict[k]

        ckpt_file_name = os.path.basename(ckpt_file)
        ckpt_file_name = ckpt_file_name[:-5]
        ckpt_path = os.path.join(output_dir, "{}-{}.ckpt".format(ckpt_file_name, config.optimize_on))
        torch.save(
            ckpt,
            ckpt_path,
        )

        print("Saved to '{}'".format(ckpt_path))


if __name__ == "__main__":
    main()
