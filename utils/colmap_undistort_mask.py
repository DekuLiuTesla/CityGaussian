"""
using colmap sparse model to undistort mask images
"""

import add_pypath
import os
import subprocess
import time
import random
import argparse
import pathlib
import numpy as np
import cv2
import tqdm
import internal.utils.colmap as read_write_model
import multiprocessing.pool

MASK_EXTENSION = "png"

parser = argparse.ArgumentParser()
parser.add_argument("--model-dir", "-m", type=str, required=True)
parser.add_argument("--mask-path", "--src", type=str, required=True)
parser.add_argument("--output-path", "--dst", type=str, default=None)
parser.add_argument("--max-size", type=int, default=-1)
args = parser.parse_args()

cameras = read_write_model.read_cameras_binary(os.path.join(args.model_dir, "cameras.bin"))
images = read_write_model.read_images_binary(os.path.join(args.model_dir, "images.bin"))

# build camera param string dict
camera_param_string_key_by_camera_id = {}
for idx in cameras:
    camera = cameras[idx]
    camera_param_string = "{} {} {} {}".format(
        camera.model,
        camera.width,
        camera.height,
        " ".join([str(i) for i in camera.params])
    )
    camera_param_string_key_by_camera_id[idx] = camera_param_string

# set output path
output_path = args.output_path
if output_path is None:
    output_path = "{}_undistorted".format(args.mask_path)

# generate a unique name for the temporary file
txt_file_path = "/tmp/image_undistorter_standalone-{}-{}.txt".format(
    int(time.time()),
    int(random.random() * 10000)
)

with open(txt_file_path, "w") as f:
    with tqdm.tqdm(images) as t:
        for idx in t:
            image = images[idx]
            # build mask name based on image name
            mask_name = f"{image.name}.{MASK_EXTENSION}"
            # check whether mask image file exists
            mask_file_path = os.path.join(args.mask_path, mask_name)
            if os.path.exists(mask_file_path) is False:
                print(f"WARNING: mask of {image.name} not found")
                continue
            # create output directory
            os.makedirs(os.path.dirname(os.path.join(args.output_path, mask_name)), exist_ok=True)
            # get camera parameter string
            camera_id = image.camera_id
            camera_param_string = camera_param_string_key_by_camera_id[camera_id]

            t.set_description(mask_name)

            # write to text file
            f.write("{} {}\n".format(
                mask_name,
                camera_param_string
            ))

print(txt_file_path)

assert subprocess.call([
    "colmap",
    "image_undistorter_standalone",
    "--input_file",
    txt_file_path,
    "--image_path",
    args.mask_path,
    "--output_path",
    output_path,
    "--max_image_size",
    str(args.max_size),
]) == 0


# the image processed by colmap will become 3 channels,
# so convert it to single channel here

def to_single_channel(filename):
    filename = str(filename)
    mask = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    if mask.shape[-1] != 3:
        return
    mask = (mask[:, :, 0] > 0).astype(np.uint8) * 255
    cv2.imwrite(filename, mask)


tasks = list(pathlib.Path(args.output_path).rglob(f"*.{MASK_EXTENSION}"))
with multiprocessing.pool.ThreadPool() as pool:
    for _ in tqdm.tqdm(pool.imap_unordered(to_single_channel, tasks), total=len(tasks)):
        pass

os.unlink(txt_file_path)
