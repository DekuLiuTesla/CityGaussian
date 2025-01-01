import add_pypath
import os
import argparse
import numpy as np
import cv2
import json
from joblib import delayed, Parallel
from internal.utils.colmap import read_model, qvec2rotmat

parser = argparse.ArgumentParser()
parser.add_argument("dataset_dir")
parser.add_argument("--depth_dir", type=str, default=None)
parser.add_argument("--output", "-o", type=str, default=None)
parser.add_argument("--point-max-error", type=float, default=1.5)
args = parser.parse_args()

if args.depth_dir is None:
    args.depth_dir = os.path.join(args.dataset_dir, "estimated_depths")
if args.output is None:
    args.output = os.path.join(args.dataset_dir, "estimated_depth_scales.json")

sparse_model_dir = os.path.join(args.dataset_dir, "sparse")
if os.path.exists(os.path.join(sparse_model_dir, "images.bin")) is False:
    sparse_model_dir = os.path.join(sparse_model_dir, "0")

cameras, images, points3d = read_model(sparse_model_dir)

# copied from https://github.com/graphdeco-inria/hierarchical-3d-gaussians/blob/main/preprocess/make_depth_scale.py

pts_indices = np.array([points3d[key].id for key in points3d])
pts_xyzs = np.array([points3d[key].xyz for key in points3d])
pts_errors = np.array([points3d[key].error for key in points3d])
points3d_ordered = np.zeros([pts_indices.max() + 1, 3])
points3d_error_ordered = np.zeros([pts_indices.max() + 1, ])
points3d_ordered[pts_indices] = pts_xyzs
points3d_error_ordered[pts_indices] = pts_errors


def get_scales(key, cameras, images, points3d_ordered, points3d_error_ordered, args):
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = images[key].point3D_ids

    # filter out invalid 3D points
    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    # get valid 3D point indices and 2D point xy
    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    # reduce outliers
    pts_errors = points3d_error_ordered[pts_idx]
    valid_errors = pts_errors < args.point_max_error
    pts_idx = pts_idx[valid_errors]
    valid_xys = valid_xys[valid_errors]

    if len(pts_idx) > 0:
        # get 3D point xyz
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    # transform from world to camera
    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2]
    invmonodepthmap = np.load(os.path.join(args.depth_dir, "{}.npy".format(image_meta.name)))  # already normalized

    if invmonodepthmap is None:
        return None

    # if invmonodepthmap.ndim != 2:
    #     invmonodepthmap = invmonodepthmap[..., 0]

    # invmonodepthmap = invmonodepthmap.astype(np.float32)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    # xys inside image
    maps = (valid_xys * s).astype(np.float32)
    valid = (
            (maps[..., 0] >= 0) *
            (maps[..., 1] >= 0) *
            (maps[..., 0] < cam_intrinsic.width * s) *
            (maps[..., 1] < cam_intrinsic.height * s) * (invcolmapdepth > 0))

    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        # depth values from colmap
        invcolmapdepth = invcolmapdepth[valid]
        # get depth values of these 2D points from the depth map
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]

        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name, "scale": scale, "offset": offset}


# depth_param_list = [get_scales(key, cameras, images, points3d_ordered, points3d_error_ordered, args) for key in images]
depth_param_list = Parallel(n_jobs=-1, backend="threading")(
    delayed(get_scales)(key, cameras, images, points3d_ordered, points3d_error_ordered, args) for key in images
)

depth_params = {
    depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
    for depth_param in depth_param_list if depth_param != None
}

with open(args.output, "w") as f:
    json.dump(depth_params, f, indent=4, ensure_ascii=False)

print("Saved to `{}`".format(args.output))
