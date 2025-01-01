import add_pypath
import os
import argparse
import torch
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.gaussian_utils import GaussianPlyUtils

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--mask", type=str, required=True)
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

assert args.model != args.output
assert args.output.endswith(".ply")
assert os.path.exists(args.output) is False

model, _ = GaussianModelLoader.search_and_load(args.model, device="cpu", eval_mode=True, pre_activate=False)
mask = torch.load(args.mask, map_location="cpu")["mask"]

properties = {key: value[mask] for key, value in model.properties.items()}
GaussianPlyUtils.load_from_model_properties(properties).to_ply_format().save_to_ply(args.output)
print(f"Saved to '{args.output}'")
