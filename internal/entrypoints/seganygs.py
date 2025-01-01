import sys
import lightning
from internal.cli import CLI
from jsonargparse import lazy_instance

from internal.segany_splatting import SegAnySplatting
from internal.dataset import DataModule
from internal.callbacks import SaveCheckpoint, ProgressBar, StopDataLoaderCacheThread
import lightning.pytorch.loggers


def cli():
    CLI(
        SegAnySplatting,
        DataModule,
        seed_everything_default=42,
        trainer_defaults={
            "accelerator": "gpu",
            "strategy": "auto",
            "devices": 1,
            # "logger": "TensorBoardLogger",
            "num_sanity_val_steps": 1,
            # "max_epochs": 100,
            "max_steps": 30_000,
            "use_distributed_sampler": False,  # use custom ddp sampler
            "enable_checkpointing": False,
            "callbacks": [
                lazy_instance(SaveCheckpoint),
                lazy_instance(ProgressBar),
                lazy_instance(StopDataLoaderCacheThread),
            ],
        },
        save_config_kwargs={"overwrite": True},
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
