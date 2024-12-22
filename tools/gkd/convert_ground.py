import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import cv2
import json
import rawpy
import xml.etree.ElementTree as ET
import numpy as np
import torch
from tqdm import tqdm
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
from internal.utils.las_utils import read_las_fit
from internal.utils.gaussian_projection import build_rotation_matrix
from internal.utils.fisheye_utils import parse_ocam_model, DeNavfisheye
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import set_start_method

import subprocess
try:
    set_start_method('spawn')
except RuntimeError:
    pass

def process_single_task(
    frame_index, 
    cam_idx, 
    sensor_frame_path,
    info_path,
    rgb_img_path,
    file_path,
    transform_parent,
    transform,
    transform_dict,
    args
):
    if not os.path.exists(rgb_img_path):
        return False
    
    info = json.load(open(info_path))
    c, d, e, cx, cy, world2cam, cam2world = parse_ocam_model(sensor_frame_path, cam_idx)   
    cam = info[f"cam{cam_idx}"]
    cam_p = cam['position']
    cam_q = cam['quaternion']
    c2w = np.eye(4)
    c2w[:3, :3] = np.array(build_rotation_matrix(torch.tensor(cam_q)[None, :]))[0]
    c2w[:3, 3] = np.array(cam_p)
    if cam_idx == 0:
        c2w[:, 0] *= -1  # cam 0
    else:
        c2w[:, 1] *= -1  # cam 1, 2, 3
    c2w = c2w[:, [1, 0, 2, 3]]
    w2c = np.linalg.inv(c2w)

    final_w2c = np.dot(w2c, np.dot(np.linalg.inv(transform), np.linalg.inv(transform_parent)))
    final_c2w = np.linalg.inv(final_w2c)
    transform_dict['frames'].append({
        "frame_index": frame_index,
        "file_path": file_path,
        "rot_mat": final_c2w.tolist()
    })
        
    if not args.skip_img_saving:
        with rawpy.imread(rgb_img_path) as raw:
            img = raw.postprocess(use_camera_wb=True, use_auto_wb=False, exp_shift=3)
            obj = DeNavfisheye(img, format='circular', pfov=args.pfov, c=c, d=d, e=e, 
                                xcenter=cx, ycenter=cy, cam2world=cam2world, world2cam=world2cam)
            new_image, f_p = obj.convert()

        os.makedirs(os.path.join(args.ground_dir_path, 'images'), exist_ok=True)
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(args.ground_dir_path, 'images', f"{frame_index:05d}.png"), new_image)
    
    return True

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Transform fisheye images and save extrinsics and intrinsics")
    parser.add_argument("--ground_dir_path", type=str, default="data/GKD/ground")
    parser.add_argument("--convert_pcd", action="store_true")
    parser.add_argument("--pcd_ds_interval", type=int, default=200)
    parser.add_argument("--skip_img_saving", action="store_true")
    parser.add_argument("--pfov", type=float, default=90)
    parser.add_argument("--max_workers", type=int, default=16)
    args = parser.parse_args(sys.argv[1:])

    if args.convert_pcd:
        print(f"Converting aerial point cloud to .ply file, downsample interval: {args.pcd_ds_interval}")
        ground_pcd_path = os.path.join(args.ground_dir_path, 'pcd')
        ground_pcd_list = sorted([os.path.join(ground_pcd_path, f) for f in os.listdir(ground_pcd_path) if f.endswith('.las')])
        xyz_ground_list, rgb_ground_list = [], []
        for ground_pcd_file in ground_pcd_list:
            xyz_ground, rgb_ground, _ = read_las_fit(ground_pcd_file, ["scales", "offsets"])
            xyz_ground_list.append(xyz_ground[::args.pcd_ds_interval])
            rgb_ground_list.append(rgb_ground[::args.pcd_ds_interval])

        xyz_ground = np.concatenate(xyz_ground_list, axis=0)
        rgb_ground = np.concatenate(rgb_ground_list, axis=0)

        # save as .ply file, note that rgb is (0, 1) float
        rgb_ground = (rgb_ground * 255).astype(np.uint8)
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        attributes = np.concatenate([xyz_ground, rgb_ground], axis=1)
        elements = np.empty(xyz_ground.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(args.ground_dir_path, 'point_cloud.ply'))
        print(f"Saved {xyz_ground.shape[0]} point cloud to {os.path.join(args.ground_dir_path, 'point_cloud.ply')}")

    transform_path = os.path.join(args.ground_dir_path, 'transforms.json')
    transform_dict = {
        "camera_angle_x": 0,
        "fl_x": 0,
        "fl_y": 0,
        "cx": 0,
        "cy": 0,
        "w": 0,
        "h": 0,
        "frames": []
    }

    input_path = os.path.join(args.ground_dir_path, 'input')
    segment_list = sorted(os.listdir(input_path))
    frame_index, intrinsics_written = 0, False

    for segment in segment_list:
        
        segment_name = segment.split('/')[-1]
        alignment = json.load(open(os.path.join(args.ground_dir_path, 'align', segment_name, 'alignment.json')))

        transform = np.array([
            [alignment['transform']['r11'], alignment['transform']['r12'], alignment['transform']['r13'], alignment['transform']['tx']],
            [alignment['transform']['r21'], alignment['transform']['r22'], alignment['transform']['r23'], alignment['transform']['ty']],
            [alignment['transform']['r31'], alignment['transform']['r32'], alignment['transform']['r33'], alignment['transform']['tz']],
            [0, 0, 0, 1]
        ])

        transform_parent = np.eye(4)
        if alignment['parent'] is not None:
            transform_parent = np.array([
                [alignment['parent']['transform']['r11'], alignment['parent']['transform']['r12'], alignment['parent']['transform']['r13'], alignment['parent']['transform']['tx']],
                [alignment['parent']['transform']['r21'], alignment['parent']['transform']['r22'], alignment['parent']['transform']['r23'], alignment['parent']['transform']['ty']],
                [alignment['parent']['transform']['r31'], alignment['parent']['transform']['r32'], alignment['parent']['transform']['r33'], alignment['parent']['transform']['tz']],
                [0, 0, 0, 1]
            ])

        sensor_frame_path = os.path.join(args.ground_dir_path, "input", segment_name, 'sensor_frame.xml')
        info_list = sorted(os.listdir(os.path.join(input_path, segment_name, 'info')))
        param_dict = {}
        for info_name in info_list:
            for cam_idx in range(4):
                file_path = os.path.join(segment_name, 'cam', f"{info_name.split('-')[0]}-cam{cam_idx}.dng")
                param_dict[frame_index] = {
                    "frame_index": frame_index,
                    "cam_idx": cam_idx,
                    "sensor_frame_path": sensor_frame_path,
                    "info_path": os.path.join(input_path, segment_name, 'info', info_name),
                    "rgb_img_path": os.path.join(input_path, file_path),
                    "file_path": file_path,
                    "transform_parent": transform_parent,
                    "transform": transform,
                    "transform_dict": transform_dict,
                    "args": args,
                }
                if not intrinsics_written:
                    intrinsics_written = True
                    with rawpy.imread(param_dict[frame_index]["rgb_img_path"]) as raw:
                        c, d, e, cx, cy, world2cam, cam2world = parse_ocam_model(sensor_frame_path, cam_idx) 
                        img = raw.postprocess(use_camera_wb=True, use_auto_wb=False, exp_shift=3)
                        obj = DeNavfisheye(img, format='circular', pfov=args.pfov, c=c, d=d, e=e, 
                                            xcenter=cx, ycenter=cy, cam2world=cam2world, world2cam=world2cam)
                        new_image, f_p = obj.convert()
                    H, W = new_image.shape[:2]
                    cx_p = (W - 1) // 2
                    cy_p = (H - 1) // 2

                    transform_dict["camera_angle_x"] = np.arctan(f_p / cx_p) * 2
                    transform_dict["fl_x"] = f_p
                    transform_dict["fl_y"] = f_p
                    transform_dict["cx"] = cx_p
                    transform_dict["cy"] = cy_p
                    transform_dict["w"] = W
                    transform_dict["h"] = H
                    
                frame_index += 1
        
        with tqdm(total=len(param_dict), desc=f"Processing {segment}") as pbar:
            with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                futures = {executor.submit(process_single_task, **param_dict[i]): i for i in list(param_dict.keys())}
                results = {}
                for future in as_completed(futures):
                    arg = futures[future]
                    results[arg] = future.result()
                    pbar.update(1)

    with open(transform_path, 'w') as f:
        # sort by frame_index
        transform_dict['frames'] = sorted(transform_dict['frames'], key=lambda x: x['frame_index'])
        json.dump(transform_dict, f, indent=4)
        print(f"Saved {frame_index} ground images' transforms to {transform_path}")
    
