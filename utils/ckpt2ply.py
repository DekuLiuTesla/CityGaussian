import add_pypath
import os
import argparse
import lightning
import torch
from internal.utils.gaussian_utils import GaussianPlyUtils
from internal.utils.gaussian_model_loader import GaussianModelLoader

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--output", "-o", required=False, default=None)
parser.add_argument("--colored", "-c", action="store_true", default=False)
args = parser.parse_args()

# search input file
print("Searching checkpoint file...")
load_file = GaussianModelLoader.search_load_file(args.input)
assert load_file.endswith(".ckpt"), f"Not a valid ckpt file can be found in '{args.input}'"

# auto select output path if not provided
if args.output is None:
    args.output = load_file[:load_file.rfind(".")] + ".ply"
    # if provided input path is a directory, write output file to `PROVIDED_PATH/point_cloud/iteration_.../point_cloud.ply`
    if os.path.isdir(args.input) is True:
        try:
            iteration = load_file[load_file.rfind("=") + 1:load_file.rfind(".")]
            if len(iteration) > 0:
                args.output = os.path.join(args.input, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
        except:
            pass

assert os.path.exists(args.output) is False, f"Output file already exists, please remove it first: '{args.output}'"

print(f"Loading checkpoint '{load_file}'...")
ckpt = torch.load(load_file)
print("Converting...")
model = GaussianPlyUtils.load_from_state_dict(ckpt["state_dict"]).to_ply_format().save_to_ply(args.output, args.colored)
print(f"Saved to '{args.output}'")

size = os.path.getsize(args.output)
size_MB = size / 1024.0 / 1024.0
print("Size = {:.2f} MB".format(size_MB))