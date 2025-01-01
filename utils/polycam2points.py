import add_pypath
import os
import argparse
import numpy as np
import torch

from tqdm.auto import tqdm
from PIL import Image
from internal.dataparsers.ngp_dataparser import NGP
from internal.utils.depth_map_utils import depth_map_to_points
from internal.utils.graphics_utils import store_ply

parser = argparse.ArgumentParser()
parser.add_argument("path")
parser.add_argument("--colmap", type=str, default=None)  # NOTE: the scale must be aligned first: `colmap model_aligner ...`
parser.add_argument("--min-confidence", "-m", type=int, default=127)
parser.add_argument("--max", type=int, default=2_048_000)
parser.add_argument("--scale", type=float, default=5.)
parser.add_argument("--max-depth", type=int, default=5000)  # The max range of the lidar sensor is 5M
args = parser.parse_args()

output_path = os.path.join(args.path, "points3D.ply")
# assert os.path.exists(output_path) is False


if args.colmap is not None:
    from internal.dataparsers.colmap_dataparser import Colmap
    dataparser = Colmap(
        image_dir=os.path.join(args.path, "keyframes", "images"),
        force_pinhole=True,
    ).instantiate(args.colmap, args.path, 0)
else:
    dataparser = NGP(
        pcd_from="random",
        num_random_points=1,
    ).instantiate(args.path, args.path, 0)

dataparser_outputs = dataparser.get_outputs()
image_set = dataparser_outputs.train_set

xyz_list = []
rgb_list = []
for image_idx in tqdm(range(len(image_set))):
    frame_basename = os.path.basename(image_set.image_names[image_idx])
    frame_basename = frame_basename.split(".")[0].split("_")[0]

    # load images
    depth = np.asarray(Image.open(os.path.join(args.path, "keyframes", "depth", "{}.png".format(frame_basename))))
    confidence = np.asarray(Image.open(os.path.join(args.path, "keyframes", "confidence", "{}.png".format(frame_basename))))
    rgb_flatten = np.asarray(Image.open(dataparser_outputs.train_set.image_paths[image_idx]).resize((depth.shape[1], depth.shape[0]))).reshape((-1, 3))

    # retrieve camera parameters
    camera = dataparser_outputs.train_set.cameras[image_idx]

    valid_depth_mask = np.logical_and(confidence > args.min_confidence, depth <= args.max_depth)
    c2w = torch.linalg.inv(camera.world_to_camera.T)

    scale_factor_x = camera.width / depth.shape[1]
    scale_factor_y = camera.height / depth.shape[0]

    # build points
    points = depth_map_to_points(
        torch.tensor(depth.astype(np.float32), dtype=torch.float) * 1e-3 * args.scale,
        fx=camera.fx / scale_factor_x,
        fy=camera.fy / scale_factor_y,
        cx=camera.cx / scale_factor_x,
        cy=camera.cy / scale_factor_y,
        c2w=c2w,
        valid_pixel_mask=valid_depth_mask
    )
    point_colors = rgb_flatten[np.reshape(valid_depth_mask, (-1,))]

    xyz_list.append(points.numpy())
    rgb_list.append(point_colors)

print("Saving...")
xyz = np.concatenate(xyz_list, axis=0)
rgb = np.concatenate(rgb_list, axis=0)
if args.max > 0 and xyz.shape[0] > args.max:
    np.random.seed(42)
    indices = np.random.permutation(xyz.shape[0])[:args.max]
    xyz = xyz[indices]
    rgb = rgb[indices]
store_ply(output_path, xyz, rgb)
