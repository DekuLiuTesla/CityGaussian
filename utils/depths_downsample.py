import numpy as np
import argparse
import cv2
import os
from os.path import isfile, join
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

args_src = None
args_dst = None
downsample_factor = None

def process_file(f):
    depth_map = np.load(join(args_src, f))
    height, width = depth_map.shape
    downsampled = cv2.resize(depth_map, (int(width // downsample_factor), int(height // downsample_factor)), interpolation=cv2.INTER_CUBIC)
    np.save(join(args_dst, f), downsampled)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("src")
    parser.add_argument("--dst", default=None)
    parser.add_argument("--factor", type=float, default=2)
    args = parser.parse_args()

    max_threads = min(32, (os.cpu_count() or 1) + 4)  

    assert args.src != args.dst

    if args.dst is None:
        args.dst = "{}_{}".format(args.src, args.factor)

    args_src = args.src
    args_dst = args.dst
    downsample_factor = args.factor
    
    print(args.dst)

    os.makedirs(args.dst)

    depth_maps = [f for f in os.listdir(args.src) if (isfile(join(args.src, f)) and f.endswith('.npy'))]

    
    with ThreadPoolExecutor(max_workers=3) as executor:
        list(tqdm(executor.map(process_file, depth_maps), total=len(depth_maps), desc="downsampling"))