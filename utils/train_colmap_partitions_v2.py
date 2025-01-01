import os
from dataclasses import dataclass
from train_partitions import PartitionTrainingConfig, PartitionTraining
import argparse


@dataclass
class ColmapPartitionTrainingConfig(PartitionTrainingConfig):
    eval: bool = False

    @classmethod
    def get_extra_init_kwargs(cls, args):
        return {
            "eval": args.eval,
        }

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        PartitionTrainingConfig.configure_argparser(parser, extra_epoches)
        parser.add_argument("--eval", action="store_true", default=False)


class ColmapPartitionTraining(PartitionTraining):
    def get_default_dataparser_name(self) -> str:
        return "Colmap"

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
    ColmapPartitionTrainingConfig.configure_argparser(parser)
    ColmapPartitionTraining.start_with_configured_argparser(parser, config_cls=ColmapPartitionTrainingConfig)


main()
