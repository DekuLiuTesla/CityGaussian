import os
from dataclasses import dataclass
from train_partitions import PartitionTrainingConfig, PartitionTraining
from distibuted_tasks import configure_arg_parser_v2
import argparse


@dataclass
class CityGSPartitionTrainingConfig(PartitionTrainingConfig):
    eval: bool = False

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        parser.add_argument("--project", "-p", type=str, required=True,
                            help="Project name")
        parser.add_argument("--min-images", "-m", type=int, default=32,
                            help="Ignore partitions with image number less than this value")
        parser.add_argument("--config", "-c", type=str, default=None)
        parser.add_argument("--quiet_wandb", "-q", type=bool, default=True)
        parser.add_argument("--parts", default=None, nargs="*", action="extend")
        parser.add_argument("--extra-epoches", "-e", type=int, default=extra_epoches)
        parser.add_argument("--scalable-config", type=str, default=None,
                            help="Load scalable params from a yaml file")
        parser.add_argument("--scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--extra-epoch-scalable-params", type=str, default=[], nargs="*", action="extend")
        parser.add_argument("--scale-param-mode", type=str, default="linear")
        parser.add_argument("--no-default-scalable", action="store_true")
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--name-suffix", type=str, default="")
        parser.add_argument("--ff-densify", action="store_true", default=False)
        configure_arg_parser_v2(parser)


class CityGSPartitionTraining(PartitionTraining):
    def get_default_dataparser_name(self) -> str:
        return "CityGS"

    def get_dataset_specific_args(self, partition_idx: int) -> list[str]:
        return [
            "--data.parser.image_list={}".format(os.path.join(
                self.path,
                "{}.txt".format(self.get_partition_id_str(partition_idx)),
            )),
            "--data.parser.split_mode={}".format("experiment" if self.config.eval else "reconstruction"),
            "--data.parser.eval_step=64",
        ]


def main():
    parser = argparse.ArgumentParser()
    CityGSPartitionTrainingConfig.configure_argparser(parser)
    CityGSPartitionTraining.start_with_configured_argparser(parser, config_cls=CityGSPartitionTrainingConfig)


main()
