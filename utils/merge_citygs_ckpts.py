import os
import sys
import add_pypath
import argparse
import numpy as np
import logging
from tqdm import tqdm
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.utils.blocking import contract_to_unisphere

parser = argparse.ArgumentParser()
parser.add_argument("path", help="Path to the model output directory")
args = parser.parse_args()

checkpoint_dir = os.path.join(args.path, "blocks")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)

file_handler = logging.FileHandler(os.path.join(os.path.dirname(checkpoint_dir), "merge.log"))
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stdout_handler)

# find max iteration
logger.info("Searching checkpoint files...")
max_iteration = -1
checkpoint_files = []
for i in os.listdir(checkpoint_dir):
    block_path = os.path.join(checkpoint_dir, i)
    loadable_file = GaussianModelLoader.search_load_file(block_path)
    if loadable_file.endswith(".ckpt") is False:
        continue
    try:
        step = int(loadable_file[loadable_file.index("step=") + 5:loadable_file.index(".ckpt")])
        if step > max_iteration:
            max_iteration = step
            checkpoint_files = []
        if step == max_iteration:
            checkpoint_files.append(loadable_file)
    except:
        pass

checkpoint_files = sorted(checkpoint_files)
assert len(checkpoint_files) > 0

logger.info(checkpoint_files)

import add_pypath
import torch
from internal.models.gaussian import Gaussian

is_new_model = True
param_list_key_by_name = {}
extra_param_list_key_by_name = {}
optimizer_state_exp_avg_list_key_by_index = {}
optimizer_state_exp_avg_sq_list_key_by_index = {}
density_controller_state_list_key_by_name = {}
number_of_gaussians = []
xyz_quantile = None
for i in tqdm(checkpoint_files, desc="Loading checkpoints"):
    ckpt = torch.load(i, map_location="cpu")
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    block_num = dataparser_config.block_dim[0] * dataparser_config.block_dim[1] * dataparser_config.block_dim[2]
    block_id_z = dataparser_config.block_id // (dataparser_config.block_dim[0] * dataparser_config.block_dim[1])
    block_id_y = (dataparser_config.block_id % (dataparser_config.block_dim[0] * dataparser_config.block_dim[1])) // dataparser_config.block_dim[0]
    block_id_x = (dataparser_config.block_id % (dataparser_config.block_dim[0] * dataparser_config.block_dim[1])) % dataparser_config.block_dim[0]
    
    if xyz_quantile is None and os.path.exists(os.path.join(os.path.dirname(dataparser_config.image_list), "quantiles.pt")):
        xyz_quantile = torch.load(os.path.join(os.path.dirname(dataparser_config.image_list), "quantiles.pt"))
        
    if xyz_quantile is None:
        # in this case, we assume partition under contracted space
        if dataparser_config.aabb is None:
            c2ws = np.array([np.linalg.inv(np.asarray((cam.world_to_camera.T).cpu().numpy())) for cam in dataset.cameras])
            poses = c2ws[:,:3,:] @ np.diag([1, -1, -1, 1])
        else:
            assert len(dataparser_config.aabb) == 6, "Unknown aabb format!"
            dataparser_config.aabb = torch.tensor(dataparser_config.aabb, dtype=torch.float32, device=ckpt['state_dict']['gaussian_model.gaussians.means'].device)
        
        min_x, max_x = float(block_id_x) / dataparser_config.block_dim[0], float(block_id_x + 1) / dataparser_config.block_dim[0]
        min_y, max_y = float(block_id_y) / dataparser_config.block_dim[1], float(block_id_y + 1) / dataparser_config.block_dim[1]
        min_z, max_z = float(block_id_z) / dataparser_config.block_dim[2], float(block_id_z + 1) / dataparser_config.block_dim[2]

        xyz_gs = contract_to_unisphere(ckpt['state_dict']['gaussian_model.gaussians.means'], dataparser_config.aabb, ord=torch.inf)
        mask_preserved = (xyz_gs[:, 0] >= min_x) & (xyz_gs[:, 0] < max_x)  \
                        & (xyz_gs[:, 1] >= min_y) & (xyz_gs[:, 1] < max_y) \
                        & (xyz_gs[:, 2] >= min_z) & (xyz_gs[:, 2] < max_z)
    else:
        x_quantile = xyz_quantile["x"]
        y_quantile = xyz_quantile["y"]
        z_quantile = xyz_quantile["z"]

        min_x, max_x = x_quantile[block_id_x].item(), x_quantile[block_id_x + 1].item()
        min_y, max_y = y_quantile[block_id_y].item(), y_quantile[block_id_y + 1].item()
        min_z, max_z = z_quantile[block_id_z].item(), z_quantile[block_id_z + 1].item()

        xyz_gs = ckpt['state_dict']['gaussian_model.gaussians.means']
        mask_preserved = (xyz_gs[:, 0] >= min_x) & (xyz_gs[:, 0] < max_x)  \
                        & (xyz_gs[:, 1] >= min_y) & (xyz_gs[:, 1] < max_y) \
                        & (xyz_gs[:, 2] >= min_z) & (xyz_gs[:, 2] < max_z)
    
    property_names = []
    gaussian_property_dict_key_prefix = "gaussian_model.gaussians."
    density_controller_state_dict_key_prefix = "density_controller."
    # extract gaussian properties
    for key, value in ckpt["state_dict"].items():
        if key.startswith(gaussian_property_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value[mask_preserved])
            property_names.append(key[len(gaussian_property_dict_key_prefix):])
        elif key.startswith(density_controller_state_dict_key_prefix):
            param_list_key_by_name.setdefault(key, []).append(value)

    # extract optimizer states, assume meet gaussian optimizers first
    for optimizer_idx, optimizer in enumerate(ckpt["optimizer_states"]):
        for param_group_idx, param_group in enumerate(optimizer["param_groups"]):
            if param_group["name"] not in property_names:
                continue

            property_names.remove(param_group["name"])
            state = optimizer["state"][param_group_idx]

            # [optimizer_idx][param_group_idx] = [state, ...]
            optimizer_state_exp_avg_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(state["exp_avg"])
            optimizer_state_exp_avg_sq_list_key_by_index.setdefault(optimizer_idx, {}).setdefault(param_group_idx, []).append(state["exp_avg_sq"])

        if len(property_names) == 0:
            break

    number_of_gaussians.append(mask_preserved.sum().item())

logger.info("Merging Gaussians and density controller states...")
ckpt["datamodule_hyper_parameters"]["parser"].block_id = None
for i in param_list_key_by_name:
    ckpt["state_dict"][i] = torch.concat(param_list_key_by_name[i], dim=0)
if is_new_model is True:
    logger.info("Merging optimizers...")
    for optimizer_idx in optimizer_state_exp_avg_list_key_by_index.keys():
        for param_group_idx in optimizer_state_exp_avg_list_key_by_index[optimizer_idx].keys():
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg"] = torch.concat(
                optimizer_state_exp_avg_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
            ckpt["optimizer_states"][optimizer_idx]["state"][param_group_idx]["exp_avg_sq"] = torch.concat(
                optimizer_state_exp_avg_sq_list_key_by_index[optimizer_idx][param_group_idx],
                dim=0,
            )
else:
    for i in extra_param_list_key_by_name:
        ckpt["gaussian_model_extra_state_dict"][i] = torch.concat(extra_param_list_key_by_name[i], dim=0)
    logger.info("Merging optimizers...")
    for i in optimizer_state_exp_avg_list_key_by_index.keys():
        ckpt["optimizer_states"][0]["state"][i]["exp_avg"] = torch.concat(optimizer_state_exp_avg_list_key_by_index[i], dim=0)
        ckpt["optimizer_states"][0]["state"][i]["exp_avg_sq"] = torch.concat(optimizer_state_exp_avg_sq_list_key_by_index[i], dim=0)


def rename_ddp_appearance_states():
    gaussian_property_dict_key_prefix = "renderer.appearance_model.module."
    for i in list(ckpt["state_dict"].keys()):
        if i.startswith(gaussian_property_dict_key_prefix) is False:
            continue
        new_key = "renderer.model.{}".format(i[len(gaussian_property_dict_key_prefix):])
        ckpt["state_dict"][new_key] = ckpt["state_dict"][i]
        del ckpt["state_dict"][i]

# TODO: align ckpt["hyper_parameters"]["renderer"] if appearance model is used

logger.info("number_of_gaussians=sum({})={}".format(number_of_gaussians, sum(number_of_gaussians)))
if not os.path.exists(os.path.join(os.path.dirname(checkpoint_dir), "checkpoints")):
    os.makedirs(os.path.join(os.path.dirname(checkpoint_dir), "checkpoints"))
output_path = os.path.join(os.path.dirname(checkpoint_dir), "checkpoints", ckpt["hyper_parameters"]["initialize_from"].split('/')[-1])
logger.info("Saving...")
torch.save(ckpt, output_path)
logger.info(f"Saved to '{output_path}'")
