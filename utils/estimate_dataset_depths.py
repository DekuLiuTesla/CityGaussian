import os
import argparse
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("dataset_path")
parser.add_argument("--image_dir", type=str, default="images")
parser.add_argument("--preview", action="store_true", default=False)
args = parser.parse_args()

assert subprocess.call(
    args=[
        "python",
        "-u",
        os.path.join(os.path.dirname(__file__), "run_depth_anything_v2.py"),
        os.path.join(args.dataset_path, args.image_dir),
    ] + (["--preview"] if args.preview else []),
    shell=False,
) == 0

assert subprocess.call(
    args=[
        "python",
        "-u",
        os.path.join(os.path.dirname(__file__), "get_depth_scales.py"),
        args.dataset_path,
    ],
    shell=False,
) == 0
