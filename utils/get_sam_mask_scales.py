import add_pypath
import os
import gc
import argparse
from tqdm import tqdm
from glob import glob
import cv2
import torch
import queue
import threading
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from internal.utils.gaussian_model_loader import GaussianModelLoader
from internal.dataparsers.colmap_dataparser import ColmapDataParser
# from internal.renderers.vanilla_depth_renderer import VanillaDepthRenderer
from internal.renderers.gsplat_contrastive_feature_renderer import GSplatContrastiveFeatureRenderer
from common import AsyncImageSaver, AsyncTensorSaver

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str)
parser.add_argument("--data_path", type=str, default=None)
parser.add_argument("--save_depth_map", "--preview", action="store_true", default=False)
args = parser.parse_args()

MODEL_DEVICE = "cuda"

# TODO: support vanilla renderer
# search checkpoint and load
load_file = GaussianModelLoader.search_load_file(args.model_path)
assert load_file.endswith(".ckpt")
model, _, ckpt = GaussianModelLoader.initialize_model_and_renderer_from_checkpoint_file(load_file, device=MODEL_DEVICE)
# renderer = VanillaDepthRenderer()
renderer = GSplatContrastiveFeatureRenderer()

# load dataset
dataset_path = ckpt["datamodule_hyper_parameters"]["path"] if args.data_path is None else args.data_path
try:
    # previous version
    dataparser_params = ckpt["datamodule_hyper_parameters"]["params"].colmap
    dataparser_params.split_mode = "reconstruction"
    dataparser = ColmapDataParser(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
        params=dataparser_params,
    )
except:
    # new version
    dataparser_config = ckpt["datamodule_hyper_parameters"]["parser"]
    setattr(dataparser_config, "split_mode", "reconstruction")  # for colmap
    dataparser = dataparser_config.instantiate(
        path=dataset_path,
        output_path=os.getcwd(),
        global_rank=0,
    )

dataparser_outputs = dataparser.get_outputs()

del ckpt
torch.cuda.empty_cache()
gc.collect()

# build output dirs
mask_dir = os.path.join(dataset_path, "semantic", "masks")
scales_dir = os.path.join(dataset_path, "semantic", "scales")
depths_dir = os.path.join(dataset_path, "semantic", "depths")
os.makedirs(scales_dir, exist_ok=True)
os.makedirs(depths_dir, exist_ok=True)


def generate_grid_index(depth):
    h, w = depth.shape
    grid = torch.meshgrid([torch.arange(h), torch.arange(w)])
    grid = torch.stack(grid, dim=-1)
    return grid


image_saver = AsyncImageSaver()
tensor_saver = AsyncTensorSaver()
try:
    bg_color = torch.zeros((3,), dtype=torch.float, device=MODEL_DEVICE)
    with tqdm(range(len(dataparser_outputs.train_set.image_paths))) as t:
        for i in t:
            # get image information
            image_path = dataparser_outputs.train_set.image_paths[i]
            image_name = dataparser_outputs.train_set.image_names[i]

            t.set_description(f"{image_name}")
            semantic_file_name = f"{image_name}.pt"

            masks = torch.load(os.path.join(mask_dir, semantic_file_name), map_location=MODEL_DEVICE)

            # get scale
            ## render depth map
            camera = dataparser_outputs.train_set.cameras[i]
            camera.to_device(MODEL_DEVICE)
            with torch.no_grad():
                depth = renderer.depth_forward(
                    camera,
                    model,
                )  # [C, H, W]
            camera.to_device("cpu")

            if args.save_depth_map is True:
                image_saver.save(
                    depth.permute(1, 2, 0).cpu().numpy(),
                    os.path.join(depths_dir, f"{image_name}.tiff")
                )

            # get 3D points in camera space
            depth = depth[0].cpu()  # [H, W]
            grid_index = generate_grid_index(depth)
            points_in_3D = torch.zeros((depth.shape[0], depth.shape[1], 3), device=MODEL_DEVICE)
            points_in_3D[:, :, -1] = depth

            cx = camera.cx.item()
            cy = camera.cy.item()
            fx = camera.fx.item()
            fy = camera.fy.item()

            points_in_3D[:, :, 0] = (grid_index[:, :, 0] - cx) * depth / fx
            points_in_3D[:, :, 1] = (grid_index[:, :, 1] - cy) * depth / fy

            # TODO: resize mask if not match to depth map
            upsampled_mask = masks.unsqueeze(1)  # [N_masks, 1, H, W]

            # count masked pixels in 3x3 rectangle
            eroded_masks = torch.conv2d(
                upsampled_mask.float(),
                torch.full((3, 3), 1.0).view(1, 1, 3, 3).to(device=upsampled_mask.device),
                padding=1,
            )
            eroded_masks = (eroded_masks >= 5).squeeze()  # (num_masks, H, W)

            scale = torch.zeros(len(masks))
            for mask_id in range(len(masks)):
                # get those points locating at masked pixels
                point_in_3D_in_mask = points_in_3D[eroded_masks[mask_id] == 1]
                # calculate the variance of the x,y and z of 3D points, then calculate its norm (length)
                scale[mask_id] = (point_in_3D_in_mask.std(dim=0) * 2).norm()

            tensor_saver.save(scale, os.path.join(scales_dir, semantic_file_name))
finally:
    image_saver.stop()
    tensor_saver.stop()
