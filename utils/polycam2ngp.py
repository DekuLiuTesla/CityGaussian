import add_pypath
import os
import sys
import argparse
import json
import subprocess
from PIL import Image
from glob import glob


parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--crop", type=int, default=5, help="Remove black border")
parser.add_argument("--min-blur", type=float, default=25., help="Ignore images with score below this value")
parser.add_argument("--scale", type=float, default=5.)
args = parser.parse_args()

KF = "keyframes"
keyframe_dir = os.path.join(args.path, KF)
previous_cwd = os.getcwd()
os.chdir(keyframe_dir)


# find frames
frame_list = sorted([os.path.basename(i).split(".")[0] for i in glob(os.path.join(keyframe_dir, "cameras", "*.json"))])
assert len(frame_list) > 0, "not a file can be found in '{}'".format(os.path.join(keyframe_dir, "cameras"))
print("{} frames found".format(len(frame_list)))

# detect dirnames
image_dir = "images"
camera_dir = "cameras"
crop_image_dir = None
if os.path.exists("corrected_cameras"):
    print("Found corrected images")
    image_dir = "corrected_images"
    camera_dir = "corrected_cameras"
    crop_image_dir = "cropped_images"
    os.makedirs(crop_image_dir, exist_ok=True)

frames = []
for i in frame_list:
    with open(os.path.join(camera_dir, "{}.json".format(i)), "r") as f:
        camera = json.load(f)
    if camera["blur_score"] < args.min_blur:
        continue

    fx = camera["fx"]
    fy = camera["fy"]
    cx = camera["cx"]
    cy = camera["cy"]
    width = camera["width"]
    height = camera["height"]

    file_load_from_dir = image_dir
    file_basename = "{}.jpg".format(i)

    if crop_image_dir is not None and args.crop > 0:
        cx -= args.crop
        cy -= args.crop
        width -= 2 * args.crop
        height -= 2 * args.crop

        file_load_from_dir = crop_image_dir

        image = Image.open(os.path.join(image_dir, file_basename))
        image = image.crop((
            args.crop,
            args.crop,
            width + args.crop,
            height + args.crop,
        ))
        image.save(os.path.join(crop_image_dir, file_basename), subsampling=0, quality=100)

    frames.append({
        "fl_x": fx,
        "fl_y": fy,
        "cx": cx,
        "cy": cy,
        "w": width,
        "h": height,
        "file_path": os.path.join(KF, file_load_from_dir, file_basename),
        "transform_matrix": [
            [camera["t_20"], camera["t_21"], camera["t_22"], camera["t_23"] * args.scale],
            [camera["t_00"], camera["t_01"], camera["t_02"], camera["t_03"] * args.scale],
            [camera["t_10"], camera["t_11"], camera["t_12"], camera["t_13"] * args.scale],
            [0.0, 0.0, 0.0, 1.0],
        ],
    })

os.chdir(previous_cwd)
print("{} frames used".format(len(frames)))
with open(os.path.join(args.path, "transforms.json"), "w") as f:
    json.dump({"frames": frames}, f, indent=4)

print("Converting depth maps to points...")
subprocess.call([
    sys.executable,
    os.path.join(os.path.dirname(__file__), "polycam2points.py"),
    args.path,
    "--scale={}".format(args.scale),
])
