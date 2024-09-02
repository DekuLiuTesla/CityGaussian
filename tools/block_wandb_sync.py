import os
import sys
import yaml
import torch
import numpy as np
import subprocess
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from concurrent.futures import ProcessPoolExecutor


def sync(wandb_path):
    cmds = [
            f"wandb sync {wandb_path}",
        ]

    for cmd in cmds:
        print(cmd)
        subprocess.run(cmd, shell=True, check=True)
    return True

def main():
    parser = ArgumentParser(description="Training script parameters")
    parser.add_argument('--output_path', type=str, help='path of output folder', default=None)
    args = parser.parse_args(sys.argv[1:])

    blocks_path = os.path.join(args.output_path, 'blocks')
    jobs = [f for f in os.listdir(blocks_path) if os.path.isdir(os.path.join(blocks_path, f))]

    with ProcessPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(sync, os.path.join(blocks_path, block, 'wandb/latest-run')) for block in jobs]

        for future in futures:
            try:
                result = future.result()
                print(f"Finished job with result: {result}\n")
            except Exception as e:
                print(f"Failed job with exception: {e}\n")

if __name__ == "__main__":
    main()