import os
import argparse
import json
import subprocess
import traceback

import torch
from tqdm import tqdm
import distibuted_tasks
from auto_hyper_parameter import auto_hyper_parameter, to_command_args

parser = argparse.ArgumentParser()
parser.add_argument("partition_path")
parser.add_argument("--model_path", "-m", type=str, required=True,
                    help="The path of the directory containing all the partitions' training outputs")
parser.add_argument("--project", "-p", type=str, required=True)
parser.add_argument("--epoch", "-e", type=int, default=30)
parser.add_argument("--exclude-parts", nargs="*", type=str, default=[], action="extend")
parser.add_argument("--parts", nargs="*", type=str, default=[], action="extend")
parser.add_argument("--dry-run", action="store_true", default=False)
parser.add_argument("--total-tasks", type=int, default=1)
parser.add_argument("--current-task-id", type=int, default=1, help="Start from 1")
args, extra_training_args = parser.parse_known_args()

args.partition_path = args.partition_path.rstrip("/")

with open(os.path.join(args.partition_path, "image_lists.json"), "r") as f:
    image_lists = json.load(f)
if len(args.exclude_parts) > 0:
    new_image_list = []
    for i in image_lists:
        if i not in args.exclude_parts:
            new_image_list.append(i)
    assert len(new_image_list) != len(image_lists)
    image_lists = new_image_list
if len(args.parts) > 0:
    new_image_list = []
    for i in image_lists:
        if i in args.parts:
            new_image_list.append(i)
    assert len(new_image_list) != len(image_lists)
    image_lists = new_image_list

# pick partitions for current process
image_lists = distibuted_tasks.get_task_list(args.total_tasks, args.current_task_id, image_lists)
with tqdm(image_lists) as t:
    for i in t:
        t.set_description(i)

        # image_list_path = os.path.join(args.partition_path, i)
        # with open(image_list_path, "rb") as f:
        #     num_images = sum(1 for _ in f)

        name = f"P_{i}"

        num_images = len(torch.load(os.path.join(args.model_path, name, "appearance_group_ids.pth"), map_location="cpu"))
        max_steps, params, scale_up = auto_hyper_parameter(num_images, extra_epoch=args.epoch)


        training_args = [
                            "python", "main.py", "fit",
                            "--config", os.path.join(args.model_path, name, "config.yaml"),
                            "--ckpt_path", os.path.join(args.model_path, name, "pruned_checkpoints", "latest-opacity_pruned-0.6.ckpt"),
                            "--logger", "wandb",
                            "--output", os.path.join("outputs", args.project),
                            "--project", args.project,
                            "--name", name,
                        ] + to_command_args(max_steps, params) + [
                            "--model.renderer.init_args.optimization.max_steps", str(max_steps),
                            # "--model.renderer.init_args.optimization.embedding_lr_init", str(0.),
                            # "--model.renderer.init_args.optimization.lr_init", str(0.),
                        ]

        if len(extra_training_args) != 0:
            training_args += extra_training_args

        try:
            if args.dry_run is False:
                subprocess.run(training_args)
            else:
                print(" ".join(training_args))
        except KeyboardInterrupt as e:
            raise e
        except:
            traceback.print_exc()
