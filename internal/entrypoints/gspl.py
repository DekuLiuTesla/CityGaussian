import sys
from internal.cli import CLI
from jsonargparse import lazy_instance
from lightning.pytorch.cli import ArgsType

from internal.gaussian_splatting import GaussianSplatting
from internal.dataset import DataModule
from internal.callbacks import SaveGaussian, KeepRunningIfWebViewerEnabled, StopImageSavingThreads, ProgressBar, ValidateOnTrainEnd, StopDataLoaderCacheThread


def cli(args: ArgsType = None):
    CLI(
        GaussianSplatting,
        DataModule,
        seed_everything_default=42,
        auto_configure_optimizers=False,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "auto",
            "devices": 1,
            # "logger": "TensorBoardLogger",
            "num_sanity_val_steps": 1,
            # "max_epochs": -1,
            "max_steps": 30_000,
            "use_distributed_sampler": False,  # use custom ddp sampler
            "enable_checkpointing": False,
            "callbacks": [
                lazy_instance(SaveGaussian),
                lazy_instance(ValidateOnTrainEnd),
                lazy_instance(KeepRunningIfWebViewerEnabled),
                lazy_instance(StopImageSavingThreads),
                lazy_instance(ProgressBar),
                lazy_instance(StopDataLoaderCacheThread),
            ],
        },
        save_config_kwargs={"overwrite": True},
        args=args,
    )
    # note: don't call fit!!


def cli_with_subcommand(subcommand: str):
    sys.argv.insert(1, subcommand)
    cli()


def cli_fit():
    cli_with_subcommand("fit")


def cli_val():
    cli_with_subcommand("validate")


def cli_test():
    cli_with_subcommand("test")


def cli_predict():
    cli_with_subcommand("predict")
