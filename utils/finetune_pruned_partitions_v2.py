import add_pypath
import os
import argparse
from typing import Dict, Any
from dataclasses import dataclass
from train_partitions import PartitionTrainingConfig, PartitionTraining


@dataclass
class PartitionFinetuningConfig(PartitionTrainingConfig):
    prune_percent: float = 0.6
    trained_project: str = None

    @classmethod
    def get_extra_init_kwargs(cls, args) -> Dict[str, Any]:
        return {
            "prune_percent": args.prune_percent,
            "trained_project": args.trained_project,
        }


class PartitionFinetuning(PartitionTraining):
    def get_overridable_partition_specific_args(self, partition_idx: int) -> list[str]:
        return super().get_overridable_partition_specific_args(partition_idx) + ["--config={}".format(os.path.join(
            self.get_project_output_dir_by_name(self.config.trained_project),
            self.get_partition_id_str(partition_idx),
            "config.yaml",
        ))]

    def get_partition_specific_args(self, partition_idx: int) -> list[str]:
        return super().get_partition_specific_args(partition_idx) + ["--ckpt_path={}".format(os.path.join(
            self.get_project_output_dir_by_name(self.config.trained_project),
            self.get_partition_id_str(partition_idx),
            "pruned_checkpoints",
            "latest-opacity_pruned-{}.ckpt".format(self.config.prune_percent)
        ))]


def main():
    parser = argparse.ArgumentParser()
    PartitionTrainingConfig.configure_argparser(parser, extra_epoches=30)
    parser.add_argument("--prune-percent", type=float, default=0.6)
    parser.add_argument("--trained-project", "-t", type=str, required=True)

    PartitionFinetuning.start_with_configured_argparser(parser, config_cls=PartitionFinetuningConfig)


main()
