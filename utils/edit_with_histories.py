import os
import add_pypath
import lightning
import argparse
import torch
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("ckpt")
parser.add_argument("history_file")
parser.add_argument("output")
args = parser.parse_args()

assert os.path.exists(args.output) is False

ckpt = torch.load(args.ckpt)
histories = torch.load(args.history_file)

xyz = ckpt["state_dict"]["gaussian_model._xyz"]
device = xyz.device
preserve_mask = torch.ones((xyz.shape[0],), dtype=torch.bool, device=device)
for operation in tqdm(histories, total=len(histories)):
    is_gaussian_selected = torch.ones(xyz.shape[0], device=xyz.device, dtype=torch.bool)
    for item in operation:
        se3, grid_size = item
        se3 = se3.to(device)
        new_xyz = torch.matmul(xyz, se3[:3, :3].T) + se3[:3, 3]
        x_mask = torch.abs(new_xyz[:, 0]) < grid_size[0] / 2
        y_mask = torch.abs(new_xyz[:, 1]) < grid_size[1] / 2
        z_mask = new_xyz[:, 2] > 0
        # update mask
        is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, x_mask)
        is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, y_mask)
        is_gaussian_selected = torch.bitwise_and(is_gaussian_selected, z_mask)
    preserve_mask = torch.bitwise_and(preserve_mask, torch.bitwise_not(is_gaussian_selected))

for i in ckpt["state_dict"]:
    if i.startswith("gaussian_model._"):
        ckpt["state_dict"][i] = ckpt["state_dict"][i][preserve_mask]
torch.save(ckpt, args.output)
print(f"{preserve_mask.sum().item()} of {preserve_mask.shape[0]} Gaussians saved to {args.output}")
