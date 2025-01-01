import add_pypath
import os
import argparse
import math
from concurrent.futures import ThreadPoolExecutor
import json
import random
import numpy as np
import torch
import cv2
from internal.utils.graphics_utils import store_ply
from internal.utils.depth_map_utils import depth_map_to_colored_points_with_down_sample
from tqdm import tqdm

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, nargs="+",
                        help="list of the path of the transforms.json")
    parser.add_argument("--output", type=str, required=True,
                        help="path to the output ply file")
    parser.add_argument("--rgb", type=str, default="rgb",
                        help="rgb image dirname")
    parser.add_argument("--scale", type=float, default=1.,
                        help="scene scale factor")
    parser.add_argument("--depth-scale", type=float, default=0.01,
                        help="depth scale factor")
    parser.add_argument("--max-points", type=int, default=20_480_000)
    parser.add_argument("--max", type=int, default=-1,
                        help="max read frames")
    parser.add_argument("--max-depth", type=float, default=65_000,
                        help="max valid depth value")
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", type=str, default="float64")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--down-sample", type=int, default=1)
    args = parser.parse_args()

    return args


def read_depth(path: str):
    return cv2.imread(
        path,
        cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
    )[..., 0]


def read_rgb(path: str):
    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def build_point_cloud(
        frames: list[dict],
        max_point_per_image: int,
        max_depth: float, scene_scale: float,
        depth_scale: float,
        rgb_dirname: str,
        dtype,
        device,
        max_workers: int,
        down_sample_factor: int,
):
    torch.inverse(torch.ones((1, 1), device=device))  # https://github.com/pytorch/pytorch/issues/90613#issuecomment-1817307008

    xyz_array_list = []
    rgb_array_list = []

    final_depth_scale = scene_scale * depth_scale
    final_max_depth = max_depth * final_depth_scale

    def build_single_frame_3d_points(frame):
        # read rgb and depth
        rgb = read_rgb(os.path.join(
            frame["base_dir"],
            rgb_dirname,
            "{:04d}.png".format(frame["frame_index"]),
        ))
        depth = torch.from_numpy(read_depth(os.path.join(
            frame["base_dir"],
            "depth",
            "{:04d}.exr".format(frame["frame_index"]),
        ))).to(dtype=dtype, device=device) * depth_scale
        valid_pixel_mask = torch.lt(depth, final_max_depth)

        # calculate intrinsics
        fov_x = frame["camera_angle_x"]
        image_shape = rgb.shape
        height, width = image_shape[0], image_shape[1]
        fx = float(.5 * width / np.tan(.5 * fov_x))
        fy = fx
        cx = width / 2
        cy = height / 2

        # build camera-to-world transform matrix
        rot_mat = torch.tensor(frame["rot_mat"], dtype=depth.dtype, device=depth.device)
        rot_mat[:3, :3] *= 100
        if scene_scale != 1.:
            rot_mat[:3, 3] *= scene_scale
        c2w = rot_mat

        """
        https://github.com/bmild/nerf/blob/18b8aebda6700ed659cb27a0c348b737a5f6ab60/run_nerf_helpers.py#L137
        https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays.html
        
        x left, y down, z front
        """
        c2w[:, 1:3] *= -1

        # build 3D points
        points_3d_in_world, rgb = depth_map_to_colored_points_with_down_sample(
            depth_map=depth,
            rgb=rgb,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            c2w=c2w,
            down_sample_factor=down_sample_factor,
            valid_pixel_mask=valid_pixel_mask,
        )

        # random sample
        n_points = points_3d_in_world.shape[0]
        if max_point_per_image < n_points:
            sample_indices = torch.randperm(n_points)[:max_point_per_image]
            points_3d_in_world = points_3d_in_world[sample_indices]
            rgb = rgb[sample_indices.cpu().numpy()]

        return points_3d_in_world.cpu(), rgb

    with ThreadPoolExecutor(max_workers=max_workers) as tpe:
        for points_3d_in_world, rgb in tqdm(
                tpe.map(build_single_frame_3d_points, frames),
                total=len(frames),
        ):
            assert points_3d_in_world.shape[0] == rgb.shape[0]
            xyz_array_list.append(points_3d_in_world)
            rgb_array_list.append(rgb)

    return torch.concat(xyz_array_list, dim=0).numpy(), np.concatenate(rgb_array_list, axis=0)


def merge_frames(paths: list[str]) -> list:
    frame_list = []
    for i in paths:
        with open(i, "r") as f:
            transforms = json.load(f)
        base_dir = os.path.dirname(i)
        fov_x = transforms["camera_angle_x"]
        for frame in transforms["frames"]:
            frame["base_dir"] = base_dir
            frame["camera_angle_x"] = fov_x
            frame_list.append(frame)

    return frame_list


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    frames = merge_frames(args.input)

    # select frames
    frames = frames[::args.step]
    if args.max > 0:
        frames = frames[:args.max]

    # calculate max point per frame
    max_point_per_image = -1
    if args.max_points > 0:
        max_point_per_image = math.ceil(args.max_points / len(frames))

    xyz, rgb = build_point_cloud(
        frames,
        max_point_per_image=max_point_per_image,
        max_depth=args.max_depth,
        scene_scale=args.scale,
        depth_scale=args.depth_scale,
        rgb_dirname=args.rgb,
        dtype=getattr(torch, args.dtype),
        device=torch.device(args.device),
        max_workers=args.max_workers,
        down_sample_factor=args.down_sample,
    )
    store_ply(args.output, xyz, rgb)


if __name__ == "__main__":
    main()
