import os.path
import internal.utils.fix_lightning_save_hyperparameters
import torch
import jsonargparse
from jsonargparse import Namespace
from jsonargparse._typehints import subclass_spec_as_namespace
from typing import Optional, Union, List, Literal
from lightning.pytorch.cli import LightningCLI, LightningArgumentParser


def discard_init_args_on_class_path_change(parser_or_action, prev_val, value):
    """
    jsonargparse will reuse args presenting in user specified instance from the default one,
    which means that parameter with same name in different class can not have different default value,
    this function prevent reusing
    """

    if prev_val and "init_args" in prev_val and prev_val["class_path"] != value["class_path"]:
        prev_val = subclass_spec_as_namespace(prev_val)
        # pop all args
        for key, val in list(prev_val.init_args.__dict__.items()):
            prev_val.init_args.pop(key)


jsonargparse._typehints.discard_init_args_on_class_path_change = discard_init_args_on_class_path_change


class CLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--max_steps", "--iterations", "--iteration", "--steps", "--step", type=Optional[int], default=None)
        parser.add_argument("--max_epochs", "--epochs", "--epoch", type=Optional[int], default=None)
        parser.add_argument("--name", "-n", type=Optional[str], default=None,
                            help="the training result output path will be 'output/name'")
        parser.add_argument("--version", "-v", type=Optional[str], default=None,
                            help="the training result output path will be 'output/name/version'")
        # TODO: add max_steps to save_iterations, but need to compatible with --max_steps < 0 & --max_epochs > 0
        parser.add_argument("--save_iterations", type=List[int], default=[7_000, 30_000])
        parser.add_argument("--logger", type=str, default="tensorboard")
        parser.add_argument("--project", type=str, default="Gaussian-Splatting", help="WanDB project name")
        parser.add_argument("--output", type=str, default=os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "outputs",
        ), help="the base directory of the output")
        parser.add_argument("--float32_matmul_precision", "-f", type=Optional[Literal["medium", "high", "highest"]], default=None)
        parser.add_argument("--viewer", action="store_true", default=False)
        parser.add_argument("--test_speed", action="store_true", default=False,
                            help="Whether to test rendering speed")
        parser.add_argument("--save_val", action="store_true", default=False,
                            help="Whether save images rendered during validation/test to files")
        parser.add_argument("--val_train", action="store_true", default=False,
                            help="Whether use train set to do validation")
        parser.add_argument("--cache_all_images", action="store_true", default=False,
                            help="Speedup validation/test by caching all images. Images in train set is cached by default.")
        parser.add_argument("--pbar_rate", type=int, default=None)

        # parser.link_arguments("iterations", "trainer.max_steps")
        # parser.link_arguments("epochs", "trainer.max_epochs")
        # parser.link_arguments("name", "logger.init_args.name")
        # parser.link_arguments("version", "logger.init_args.version")
        # parser.link_arguments("output", "logger.init_args.save_dir")
        # parser.link_arguments("output", "model.output_dir")
        # parser.link_arguments("logger", "trainer.logger", apply_on="instantiate")
        parser.link_arguments("save_iterations", "model.save_iterations")

    def _search_checkpoint(self, path: str) -> str:
        from internal.utils.gaussian_model_loader import GaussianModelLoader
        ckpt_path = GaussianModelLoader.search_load_file(path)
        assert ckpt_path.endswith(".ckpt"), "not a checkpoint can be found in {}".format(path)
        print("Auto select checkpoint file: {}".format(ckpt_path))
        return ckpt_path

    def before_instantiate_classes(self) -> None:
        config = getattr(self.config, self.config.subcommand)
        if config.name is None:
            # auto set experiment name base on --data.path
            config.name = "_".join(config.data.path.strip("/").split("/")[-3:])
            print("auto determine experiment name: {}".format(config.name))

        if config.max_steps is not None:
            config.trainer.max_steps = config.max_steps
        if config.max_epochs is not None:
            config.trainer.max_epochs = config.max_epochs

        # build output path
        output_path = os.path.join(config.output, config.name)
        if config.version is not None:
            output_path = os.path.join(output_path, config.version)
        if "block_id" in config.data.parser.init_args and config.data.parser.init_args.block_id is not None:
            output_path = os.path.join(output_path, "blocks", "block_{}".format(config.data.parser.init_args.block_id))
            # config.model.init_from = config.model.init_from.replace(".ckpt", f"_block_{params.block_id}.ckpt")
        os.makedirs(output_path, exist_ok=True)
        print("output path: {}".format(output_path))
        config.model.output_path = output_path

        # search checkpoint
        if config.ckpt_path == "last":
            config.ckpt_path = self._search_checkpoint(output_path)

        if self.config.subcommand == "fit":
            if config.ckpt_path is None:
                assert (os.path.exists(
                    os.path.join(output_path, "point_cloud")
                ) or os.path.exists(
                    os.path.join(output_path, "checkpoints")
                )) is False, ("checkpoint or point cloud output already exists in '{}', \n"
                              "please specific a different experiment name (-n) or version (-v)").format(output_path)
        else:
            # disable logger
            config.logger = "None"
            # disable config saveing
            self.save_config_callback = None
            # find checkpoint automatically if not provided
            if config.ckpt_path is None:
                config.ckpt_path = self._search_checkpoint(output_path)

        # build logger
        logger_config = Namespace(
            class_path=None,
            init_args=Namespace(
                save_dir=output_path,
            ),
        )

        if config.logger == "tensorboard":
            logger_config.class_path = "lightning.pytorch.loggers.TensorBoardLogger"
        elif config.logger == "wandb":
            logger_config.class_path = "lightning.pytorch.loggers.WandbLogger"
            wandb_name = config.name
            if config.version is not None:
                wandb_name = "{}_{}".format(wandb_name, config.version)
            if config.data.parser.init_args.block_id is not None:
                wandb_name = "{}_block_{}".format(wandb_name, config.data.parser.init_args.block_id)
            setattr(logger_config.init_args, "name", wandb_name)
            setattr(logger_config.init_args, "project", config.project)
        elif config.logger == "none" or config.logger == "None" or config.logger == "false" or config.logger == "False":
            logger_config = False
        else:
            logger_config.class_path = config.logger

        config.trainer.logger = logger_config

        # set torch float32_matmul_precision
        if config.float32_matmul_precision is not None:
            torch.set_float32_matmul_precision(config.float32_matmul_precision)

        # set web viewer
        config.model.web_viewer = config.viewer

        config.model.test_speed = config.test_speed
        config.model.save_val_output = config.save_val
        config.data.val_on_train = config.val_train

        # set number of cached images
        if config.cache_all_images is True:
            config.data.train_max_num_images_to_cache = -1
            config.data.val_max_num_images_to_cache = -1
            config.data.test_max_num_images_to_cache = -1

        # set refresh rate of the progress bar
        if config.pbar_rate is not None:
            for i in self.trainer_defaults["callbacks"]:
                if i.__class__.__name__ == "LazyInstance_ProgressBar":
                    i._lazy_kwargs["refresh_rate"] = config.pbar_rate
