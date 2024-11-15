import add_pypath
import os
import argparse
import lightning
import torch
from internal.utils.gaussian_utils import Gaussian
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.vq_utils import read_ply_data, write_ply_data, load_vqgaussian

parser = argparse.ArgumentParser()
parser.add_argument("input")
parser.add_argument("--reference", "-r", required=True, default=None)
parser.add_argument("--output", "-o", required=False, default=None)
parser.add_argument("--sh_degree", "-s", required=False, default=3, type=int)
args = parser.parse_args()

# search input file
print("Searching checkpoint file...")
load_file = GaussianModelLoader.search_load_file(args.reference)
assert load_file.endswith(".ckpt"), f"Not a valid reference ckpt file can be found in '{args.reference}'"

if args.sh_degree == 3:
    sh_dim = 3+45
elif args.sh_degree == 2:
    sh_dim = 3+24

print(f"Loading checkpoint '{load_file}'...")
checkpoint = torch.load(load_file)
print("Converting...")
if os.path.exists(os.path.join(args.input, 'extreme_saving')):
    dequantized_feats = load_vqgaussian(os.path.join(args.input,'extreme_saving'), device='cuda')
    checkpoint["state_dict"]["gaussian_model._xyz"] = dequantized_feats[:, :3]
    checkpoint["state_dict"]["gaussian_model._features_dc"] = dequantized_feats[:, 6:9].reshape(dequantized_feats.shape[0], 1, 3)
    checkpoint["state_dict"]["gaussian_model._features_rest"] = dequantized_feats[:, 9:6+sh_dim].reshape(dequantized_feats.shape[0], -1, 3)
    checkpoint["state_dict"]["gaussian_model._opacity"] = dequantized_feats[:, -8:-7]
    checkpoint["state_dict"]["gaussian_model._scaling"] = dequantized_feats[:, -7:-4]
    checkpoint["state_dict"]["gaussian_model._rotation"] = dequantized_feats[:, -4:]
    checkpoint["state_dict"]["gaussian_model._features_extra"] = torch.empty(dequantized_feats.shape[0], 0).to('cuda')
    checkpoint["gaussian_model_extra_state_dict"]["active_sh_degree"] = args.sh_degree
    if args.output is None:
        args.output = os.path.join(args.input, 'checkpoints', args.reference.split("/")[-1])
    
elif args.input.endswith(".ply"):
    model = Gaussian.load_from_ply(args.input, sh_degrees=args.sh_degree).to_parameter_structure()
    checkpoint["state_dict"]["gaussian_model._xyz"] = model.xyz
    checkpoint["state_dict"]["gaussian_model._opacity"] = model.opacities
    checkpoint["state_dict"]["gaussian_model._features_dc"] = model.features_dc
    checkpoint["state_dict"]["gaussian_model._features_rest"] = model.features_rest
    checkpoint["state_dict"]["gaussian_model._scaling"] = model.scales
    checkpoint["state_dict"]["gaussian_model._rotation"] = model.rotations
    checkpoint["state_dict"]["gaussian_model._features_extra"] = model.real_features_extra
    checkpoint["gaussian_model_extra_state_dict"]["active_sh_degree"] = args.sh_degree
    if args.output is None:
        os.makedirs(os.path.join(args.input[:args.input.rfind("/")], 'checkpoints'), exist_ok=True)
        args.output = os.path.join(args.input[:args.input.rfind("/")], 'checkpoints', args.reference.split("/")[-1])
torch.save(checkpoint, args.output)

print(f"Saved to '{args.output}'")
