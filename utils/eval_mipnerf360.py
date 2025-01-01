import os
import subprocess
import argparse
import distibuted_tasks
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--config", "-c", default=None)
parser.add_argument("--down_sample_factor", "--down-sample-facotr", "-d", type=int, default=4)
parser.add_argument("--project", "-p", default="MipNeRF360")
distibuted_tasks.configure_arg_parser(parser)
args, fitting_args = parser.parse_known_args()
print(fitting_args)

# find scenes
scenes = []
for i in list(os.listdir(args.path)):
    if os.path.isdir(os.path.join(args.path, i, "sparse")) is False:
        continue
    scenes.append(i)
scenes.sort()
scenes = distibuted_tasks.get_task_list_with_args(args, scenes)
print(scenes)


def start(command: str, scene: str, extra_args: list = None):
    arg_list = [
        "python",
        "main.py",
        command,
        "--data.parser", "Colmap",  # this can be overridden by config file or args latter
    ]
    if args.config is not None:
        arg_list += ["--config", args.config]
    if extra_args is not None:
        arg_list += extra_args
    arg_list += [
        "--data.path", os.path.join(args.path, scene),
        "--data.parser.down_sample_factor", "{}".format(args.down_sample_factor),
        "--data.parser.split_mode", "experiment",
        "--data.parser.down_sample_rounding_mode", "round_half_up",
        "--cache_all_images",
        "--logger", "wandb",
        "--output", os.path.join("outputs", args.project),
        "--project", args.project,
        "-n", scene,
    ]

    return subprocess.call(arg_list)


with tqdm(scenes) as t:
    for i in t:
        t.set_description(i)
        start("fit", i, extra_args=fitting_args)
        start("validate", i, extra_args=fitting_args + ["--save_val"])
