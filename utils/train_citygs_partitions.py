import os
import argparse
import yaml
import concurrent
import add_pypath
import subprocess
import traceback
import time
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from auto_hyper_parameter import auto_hyper_parameter, to_command_args
from internal.utils.general_utils import parse

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

parser = argparse.ArgumentParser()
parser.add_argument("--config_name", "-n", type=str, required=True)
parser.add_argument("--config_dir", "-c", type=str, default="./configs")
parser.add_argument("--project", "-p", type=str, required=True)
parser.add_argument("--dry-run", action="store_true", default=False)
parser.add_argument("--enable_slurm", action="store_true", default=False)
args, extra_training_args = parser.parse_known_args()

config_path = os.path.join(args.config_dir, f"{args.config_name}.yaml")
with open(config_path, 'r') as f:
    config = parse(yaml.load(f, Loader=yaml.FullLoader))

assert os.path.exists(config.data.parser.init_args.image_list), "Image list not found, please generate it with utils/partition_citygs.py"
num_blocks = len(os.listdir(config.data.parser.init_args.image_list))

if not args.enable_slurm:
    with tqdm(range(num_blocks)) as t:
        for block_idx in t:
            train_a_partition(block_idx=block_idx)
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

print('here')