import os
import argparse
import json
import subprocess
from tqdm.auto import tqdm
from auto_hyper_parameter import auto_hyper_parameter, to_command_args

parser = argparse.ArgumentParser()
parser.add_argument("path", help="The directory where the partition data placed, "
                                 "e.g.: `data/LargeScene/colmap/dense_max_2048/0/partitions-threshold_0.2`")
parser.add_argument("--config", "-c", type=str, required=True)
parser.add_argument("--project", "-p", type=str, required=True)
parser.add_argument("--parts", "-b", type=str, nargs="*", default=[])
parser.add_argument("--dry-run", action="store_true", default=False)
parser.add_argument("--total-tasks", type=int, default=1)
parser.add_argument("--current-task-id", type=int, default=1, help="Start from 1")
args, extra_training_args = parser.parse_known_args()

args.path = args.path.rstrip("/")

assert args.total_tasks > 0
assert args.current_task_id > 0 and args.current_task_id <= args.total_tasks

with open(os.path.join(args.path, "image_lists.json"), "r") as f:
    image_lists = json.load(f)

# pick partitions for current process
num_tasks_per_process = int(round(len(image_lists) / args.total_tasks))
slice_start = (args.current_task_id - 1) * num_tasks_per_process
slice_end = slice_start + num_tasks_per_process
if args.current_task_id == args.total_tasks:
    slice_end = len(image_lists)
image_lists = image_lists[slice_start:slice_end]

with tqdm(image_lists) as t:
    for i in t:
        t.set_description(i)
        if len(args.parts) > 0 and i not in args.parts:
            continue

        try:
            image_list_path = os.path.join(args.path, i)
            with open(image_list_path, "rb") as f:
                num_images = sum(1 for _ in f)

            max_steps, params, scale_up = auto_hyper_parameter(num_images)

            training_args = [
                                "python", "main.py", "fit",
                                "--data.parser", "Colmap",
                                "--config", f"configs/{args.config}.yaml",
                                "--data.path", os.path.dirname(args.path),
                                "--data.parser.appearance_groups", "appearance_image_dedicated",
                                "--data.parser.image_list", image_list_path,
                                "--data.parser.eval_step", "64",
                                "--logger", "wandb",
                                "--output", os.path.join("outputs", args.project),
                                "--project", args.project,
                                "--name", f"P_{i}",
                            ] + to_command_args(max_steps, params) + [
                                "--model.renderer.init_args.optimization.max_steps",
                                str(max_steps),
                            ]
            if len(extra_training_args):
                training_args += extra_training_args
            if args.dry_run is False:
                subprocess.run(training_args)
            else:
                print(" ".join(training_args))
        except:
            pass
