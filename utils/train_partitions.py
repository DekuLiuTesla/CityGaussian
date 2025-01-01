import concurrent.futures

import add_pypath
from typing import Union, Optional, Literal, List, Tuple, Dict, Any
import os
import time
import traceback
import yaml
import torch
import subprocess
import selectors
from dataclasses import dataclass, field
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from internal.utils.partitioning_utils import PartitionCoordinates
from distibuted_tasks import get_task_list
from auto_hyper_parameter import auto_hyper_parameter, to_command_args, SCALABEL_PARAMS, EXTRA_EPOCH_SCALABLE_STEP_PARAMS
from argparser_utils import split_stoppable_args, parser_stoppable_args
from distibuted_tasks import configure_arg_parser_v2


@dataclass
class PartitionTrainingConfig:
    partition_dir: str
    project_name: str
    min_images: int
    n_processes: int
    process_id: int
    dry_run: bool
    extra_epoches: int
    name_suffix: str = ""
    ff_densify: bool = False
    scalable_params: Optional[Dict[str, int]] = None
    extra_epoch_scalable_params: Optional[List[str]] = None
    scale_param_mode: Literal["linear", "sqrt", "none"] = "linear"
    partition_id_strs: Optional[List[str]] = None
    training_args: Union[Tuple, List] = None
    config_file: Optional[str] = None
    srun_args: List[str] = field(default_factory=lambda: [])

    def __post_init__(self):
        if self.scalable_params is None:
            self.scalable_params = {}
        if self.extra_epoch_scalable_params is None:
            self.extra_epoch_scalable_params = []
        if self.training_args is None:
            self.training_args = []

        self.training_args = tuple(self.training_args)

    @staticmethod
    def configure_argparser(parser, extra_epoches: int = 0):
        # TODO: replace with jsonargparse
        parser.add_argument("partition_dir")
        parser.add_argument("--project", "-p", type=str, required=True,
                            help="Project name")
        parser.add_argument("--min-images", "-m", type=int, default=32,
                            help="Ignore partitions with image number less than this value")
        parser.add_argument("--config", "-c", type=str, default=None)
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

    @staticmethod
    def parse_scalable_params(args):
        # parse scalable params
        scalable_params = SCALABEL_PARAMS
        extra_epoch_scalable_params = EXTRA_EPOCH_SCALABLE_STEP_PARAMS
        if args.no_default_scalable:
            scalable_params = {}
            extra_epoch_scalable_params = []

        if args.scalable_config is not None:
            with open(args.scalable_config, "r") as f:
                scalable_config = yaml.safe_load(f)

            for i in scalable_config.keys():
                if i not in ["scalable", "extra_epoch_scalable", "no_default"]:
                    raise ValueError("Found an unexpected key '{}' in '{}'".format(i, args.scalable_yaml))

            if scalable_config.get("no_default", False):
                scalable_params = {}
                extra_epoch_scalable_params = []

            scalable_params.update(scalable_config.get("scalable", {}))
            extra_epoch_scalable_params += scalable_config.get("extra_epoch_scalable", [])

        for i in args.scalable_params:
            name, value = i.split("=", 1)
            value = int(value)
            scalable_params[name] = value
        extra_epoch_scalable_params += args.extra_epoch_scalable_params

        return scalable_params, extra_epoch_scalable_params

    @classmethod
    def get_extra_init_kwargs(cls, args) -> Dict[str, Any]:
        return {}

    @classmethod
    def instantiate_with_args(cls, args, training_args, srun_args):
        scalable_params, extra_epoch_scalable_params = cls.parse_scalable_params(args)

        return cls(
            partition_dir=args.partition_dir,
            project_name=args.project,
            min_images=args.min_images,
            n_processes=args.n_processes,
            process_id=args.process_id,
            dry_run=args.dry_run,
            extra_epoches=args.extra_epoches,
            name_suffix=args.name_suffix,
            ff_densify=args.ff_densify,
            scalable_params=scalable_params,
            extra_epoch_scalable_params=extra_epoch_scalable_params,
            scale_param_mode=args.scale_param_mode,
            partition_id_strs=args.parts,
            config_file=args.config,
            training_args=training_args,
            srun_args=srun_args,
            **cls.get_extra_init_kwargs(args),
        )

    @classmethod
    def instantiate_with_parser(cls, parser):
        args, training_and_srun_args = parser_stoppable_args(parser)
        training_args, srun_args = split_stoppable_args(training_and_srun_args)
        return cls.instantiate_with_args(args, training_args, srun_args), args


class PartitionTraining:
    def __init__(
            self,
            config: PartitionTrainingConfig,
            name: str = "partitions.pt",
    ):
        self.path = config.partition_dir
        self.config = config
        self.scene = torch.load(os.path.join(self.path, name), map_location="cpu")
        self.scene["partition_coordinates"] = PartitionCoordinates(**self.scene["partition_coordinates"])
        self.dataset_path = os.path.dirname(self.path.rstrip("/"))

    @property
    def partition_coordinates(self) -> PartitionCoordinates:
        return self.scene["partition_coordinates"]

    @staticmethod
    def get_project_output_dir_by_name(project_name: str) -> str:
        return os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__))), "outputs", project_name)

    @property
    def project_output_dir(self) -> str:
        return self.get_project_output_dir_by_name(self.config.project_name)

    @property
    def srun_output_dir(self) -> str:
        return os.path.join(self.project_output_dir, "srun-outputs")

    def run_subprocess(self, args, output_redirect) -> int:
        sel = selectors.DefaultSelector()

        with subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
            sel.register(p.stdout, selectors.EVENT_READ)
            sel.register(p.stderr, selectors.EVENT_READ)

            while True:
                if len(sel.get_map()) == 0:
                    break

                events = sel.select()
                for key, mask in events:
                    line = key.fileobj.readline()
                    if len(line) == 0:
                        sel.unregister(key.fileobj)
                        continue
                    output_redirect(line.decode("utf-8").rstrip("\n"))
            p.wait()
            return p.returncode

    def get_default_dataparser_name(self) -> str:
        raise NotImplementedError()

    def get_dataset_specific_args(self, partition_idx: int) -> list[str]:
        return []

    def get_overridable_partition_specific_args(self, partition_idx: int) -> list[str]:
        args = []
        if self.config.ff_densify:
            args += [
                "--model.density=internal.density_controllers.foreground_first_density_controller.ForegroundFirstDensityController",
                "--model.density.partition={}".format(self.path),
                "--model.density.partition_idx={}".format(partition_idx),
            ]
        return args

    def get_partition_specific_args(self, partition_idx: int) -> list[str]:
        return []

    def get_location_based_assignment_numbers(self) -> torch.Tensor:
        return self.scene["location_based_assignments"].sum(-1)

    def get_partition_id_str(self, idx: int) -> str:
        return self.partition_coordinates.get_str_id(idx)

    def get_image_numbers(self) -> torch.Tensor:
        return torch.logical_or(
            self.scene["location_based_assignments"],
            self.scene["visibility_based_assignments"],
        ).sum(-1)

    def get_trainable_partition_idx_list(
            self,
            min_images: int,
            n_processes: int,
            process_id: int,
    ) -> list[int]:
        location_based_numbers = self.get_location_based_assignment_numbers()
        all_trainable_partition_indices = torch.ge(location_based_numbers, min_images).nonzero().squeeze(-1).tolist()
        return get_task_list(n_processors=n_processes, current_processor_id=process_id, all_tasks=all_trainable_partition_indices)

    def get_partition_trained_step_filename(self, partition_idx: int):
        return "{}-trained".format(self.get_experiment_name(partition_idx))

    def get_partition_image_number(self, partition_idx: int) -> int:
        return torch.logical_or(
            self.scene["location_based_assignments"][partition_idx],
            self.scene["visibility_based_assignments"][partition_idx],
        ).sum(-1).item()

    def get_experiment_name(self, partition_idx: int) -> str:
        return "{}{}".format(self.get_partition_id_str(partition_idx), self.config.name_suffix)

    def train_a_partition(
            self,
            partition_idx: int,
    ):
        partition_image_number = self.get_partition_image_number(partition_idx)
        extra_epoches = self.config.extra_epoches
        scalable_params = self.config.scalable_params
        extra_epoch_scalable_params = self.config.extra_epoch_scalable_params
        project_output_dir = self.project_output_dir
        config_file = self.config.config_file
        extra_training_args = self.config.training_args
        project_name = self.config.project_name
        dry_run = self.config.dry_run

        # scale hyper parameters
        max_steps, scaled_params, scale_up = auto_hyper_parameter(
            partition_image_number,
            extra_epoch=extra_epoches,
            scalable_params=scalable_params,
            extra_epoch_scalable_params=extra_epoch_scalable_params,
            scale_mode=self.config.scale_param_mode,
        )

        # whether a trained partition
        partition_trained_step_file_path = os.path.join(
            project_output_dir,
            self.get_partition_trained_step_filename(partition_idx)
        )

        try:
            with open(partition_trained_step_file_path, "r") as f:
                trained_steps = int(f.read())
                if trained_steps >= max_steps:
                    print("Skip trained partition '{}'".format(self.partition_coordinates.id[partition_idx].tolist()))
                    return partition_idx, 0
        except:
            pass

        partition_output_dir = os.path.join(project_output_dir, self.get_experiment_name(partition_idx))
        if os.path.exists(partition_output_dir):
            previous_output_new_dir = "{}-{}".format(partition_output_dir, int(time.time()))
            print("Move existing {} to {}".format(partition_output_dir, previous_output_new_dir))
            os.rename(partition_output_dir, previous_output_new_dir)

        # build args
        # basic
        args = [
            "python",
            "main.py", "fit",
        ]
        # dataparser; finetune does not require setting `--data.parser`
        try:
            args += ["--data.parser", self.get_default_dataparser_name()]  # can be overridden by config file or the args later
        except NotImplementedError:
            pass

        # config file
        if config_file is not None:
            args.append("--config={}".format(config_file))

        args += self.get_overridable_partition_specific_args(partition_idx)

        # extra
        args += extra_training_args

        # dataset specified
        args += self.get_dataset_specific_args(partition_idx)

        # scalable
        args += to_command_args(max_steps, scaled_params)

        experiment_name = self.get_experiment_name(partition_idx)
        args += [
            "-n={}".format(experiment_name),
            "--data.path", self.dataset_path,
            "--project", project_name,
            "--output", project_output_dir,
            "--logger", "wandb",
        ]

        args += self.get_partition_specific_args(partition_idx)

        print_func = print
        run_func = subprocess.call
        if len(self.config.srun_args) > 0:
            def tqdm_write(i):
                tqdm.write("[{}] #{}({}): {}".format(
                    time.strftime('%Y-%m-%d %H:%M:%S'),
                    partition_idx,
                    self.get_partition_id_str(partition_idx),
                    i,
                ))

            def run_with_tqdm_write(args):
                return self.run_subprocess(args, tqdm_write)

            run_func = run_with_tqdm_write
            print_func = tqdm_write

            output_filename = os.path.join(self.srun_output_dir, "{}.txt".format(experiment_name))
            args = [
                "srun",
                "--output={}".format(output_filename),
                "--job-name={}-{}".format(self.config.project_name, experiment_name),
            ] + self.config.srun_args + args

        ret_code = -1
        if dry_run:
            print(" \\\n  ".join(args))
        else:
            try:
                print_func(str(args))
                ret_code = run_func(args)
                if ret_code == 0:
                    with open(partition_trained_step_file_path, "w") as f:
                        f.write("{}".format(max_steps))
            except KeyboardInterrupt as e:
                raise e
            except:
                traceback.print_exc()

        return partition_idx, ret_code

    def train_partitions(
            self,
    ):
        raw_trainable_partition_idx_list = self.get_trainable_partition_idx_list(
            min_images=self.config.min_images,
            n_processes=self.config.n_processes,
            process_id=self.config.process_id,
        )
        print([self.get_partition_id_str(i) for i in raw_trainable_partition_idx_list])

        trainable_partition_idx_list = []
        for partition_idx in raw_trainable_partition_idx_list:
            partition_id_str = self.get_partition_id_str(partition_idx)
            if self.config.partition_id_strs is not None and partition_id_str not in self.config.partition_id_strs:
                continue
            trainable_partition_idx_list.append(partition_idx)

        if len(self.config.srun_args) == 0:
            with tqdm(trainable_partition_idx_list) as t:
                for partition_idx in t:
                    self.train_a_partition(partition_idx=partition_idx)
        else:
            print("SLURM mode enabled")
            os.makedirs(self.srun_output_dir, exist_ok=True)
            print("Running outputs will be saved to '{}'".format(self.srun_output_dir))
            total_trainable_partitions = len(trainable_partition_idx_list)

            with ThreadPoolExecutor(max_workers=total_trainable_partitions) as tpe:
                futures = [tpe.submit(
                    self.train_a_partition,
                    i,
                ) for i in trainable_partition_idx_list]
                finished_count = 0
                with tqdm(
                        concurrent.futures.as_completed(futures),
                        total=total_trainable_partitions,
                        miniters=1,
                        mininterval=0,  # keep progress bar updating
                        maxinterval=0,
                ) as t:
                    for future in t:
                        finished_count += 1
                        try:
                            finished_idx, ret_code = future.result()
                        except KeyboardInterrupt as e:
                            raise e
                        except:
                            traceback.print_exc()
                            continue
                        tqdm.write("[{}] #{}({}) exited with code {} | {}/{}".format(
                            time.strftime('%Y-%m-%d %H:%M:%S'),
                            finished_idx,
                            self.get_partition_id_str(finished_idx),
                            ret_code,
                            finished_count,
                            total_trainable_partitions,
                        ))

    @classmethod
    def start_with_configured_argparser(cls, parser, config_cls=PartitionTrainingConfig):
        config, args = config_cls.instantiate_with_parser(parser)

        partition_training = cls(config)

        # start training
        partition_training.train_partitions()
