# -*- coding:utf8 -*-
import cv2
import os
import shutil
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from tqdm import tqdm
from argparse import ArgumentParser
from internal.utils.colmap import *

def merge_data(matrix_s2t, source_path, target_path, output_path):

    src_pcd_path = os.path.join(source_path, "sparse/0/points3D.bin")
    src_img_path = os.path.join(source_path, "sparse/0/images.bin")
    src_intr_path = os.path.join(source_path, "sparse/0/cameras.bin")
    src_images_folder = os.path.join(source_path, "images")

    tgt_pcd_path = os.path.join(target_path, "sparse/0/points3D.bin")
    tgt_img_path = os.path.join(target_path, "sparse/0/images.bin")
    tgt_intr_path = os.path.join(target_path, "sparse/0/cameras.bin")
    tgt_images_folder = os.path.join(target_path, "images")

    out_pcd_path = os.path.join(output_path, "sparse/0/points3D.bin")
    out_img_path = os.path.join(output_path, "sparse/0/images.bin")
    out_cam_path = os.path.join(output_path, "sparse/0/cameras.bin")
    out_images_folder = os.path.join(output_path, "images")
    if not os.path.exists(os.path.join(output_path, "sparse/0")):
        os.makedirs(os.path.join(output_path, "sparse/0"))
    if not os.path.exists(out_images_folder):
        os.makedirs(out_images_folder)

    # soft link images from source and target to output folder
    # here we assumed images from source and target have different names
    if not os.path.exists(out_images_folder):
        os.makedirs(out_images_folder)
    
    for img in os.listdir(src_images_folder):
        os.symlink(os.path.abspath(os.path.join(src_images_folder, img)), os.path.abspath(os.path.join(out_images_folder, img)))
    # for img in os.listdir(tgt_images_folder):
    #     os.symlink(os.path.abspath(os.path.join(tgt_images_folder, img)), os.path.abspath(os.path.join(out_images_folder, img)))
    
    # merge cameras.bin, src camera as id 2, tgt camera as id 1
    src_intr = read_cameras_binary(src_intr_path)
    tgt_intr = read_cameras_binary(tgt_intr_path)
    # src_id = 2
    # tgt_id = 1
    # cameras = {
    #     1: Camera(id=tgt_id, model=tgt_intr[1].model, width=tgt_intr[1].width, height=tgt_intr[1].height, params=tgt_intr[1].params),
    #     2: Camera(id=src_id, model=src_intr[1].model, width=src_intr[1].width, height=src_intr[1].height, params=src_intr[1].params)
    # }
    src_id = 1
    cameras = src_intr
    write_cameras_binary(cameras, out_cam_path)
    print(f"cameras.bin saved to {out_cam_path}")

    src_extr = read_images_binary(src_img_path)
    tgt_extr = read_images_binary(tgt_img_path)
    src_pcd = read_points3D_binary(src_pcd_path)
    tgt_pcd = read_points3D_binary(tgt_pcd_path)
    src_cam_start_idx = max(list(tgt_extr.keys())) + 1
    src_pcd_start_idx = max(list(tgt_pcd.keys())) + 1
    src_cam_idx_mapping = {cam.id: cam.id + src_cam_start_idx for cam in src_extr.values()}
    src_pcd_idx_mapping = {pcd.id: pcd.id + src_pcd_start_idx for pcd in src_pcd.values()}

    # merge points3D.bin
    # output_points = tgt_pcd.copy()
    output_points = {}
    for pcd in src_pcd.values():
        homo_xyz = np.ones(4)
        homo_xyz[:3] = pcd.xyz
        transformed_xyz = matrix_s2t.dot(homo_xyz)
        output_points[src_pcd_idx_mapping[pcd.id]] = Point3D(
            id=src_pcd_idx_mapping[pcd.id],
            xyz=transformed_xyz[:3],
            rgb=pcd.rgb,
            error=pcd.error,
            image_ids=np.array([src_cam_idx_mapping[img_id] for img_id in pcd.image_ids]),
            point2D_idxs=pcd.point2D_idxs
        )
    write_points3D_binary(output_points, out_pcd_path)
    print(f"points3D.bin saved to {out_pcd_path}")

    # merge images.bin
    # output_extr = tgt_extr.copy()
    output_extr = {}
    matrix_t2s = np.linalg.inv(matrix_s2t)
    for cam in src_extr.values():
        R_src_cam = qvec2rotmat(cam.qvec)
        T_src_cam = np.array(cam.tvec)
        w2c_src_cam = np.identity(4)
        w2c_src_cam[:3, :3] = R_src_cam
        w2c_src_cam[:3, 3] = T_src_cam
        w2c_transformed = w2c_src_cam.dot(matrix_t2s)

        output_extr[src_cam_idx_mapping[cam.id]] = BaseImage(
            camera_id=src_id,
            id=src_cam_idx_mapping[cam.id],
            name=cam.name,
            point3D_ids=np.array([src_pcd_idx_mapping[pcd_id] if pcd_id>=0 else pcd_id for pcd_id in cam.point3D_ids]),
            qvec=rotmat2qvec(w2c_transformed[:3, :3]),
            tvec=w2c_transformed[:3, 3] / np.linalg.norm(w2c_transformed[0, :3]),
            xys=cam.xys,
        )
    write_images_binary(output_extr, out_img_path)
    print(f"images.bin saved to {out_img_path}")

 
if __name__ == '__main__':
    parser = ArgumentParser(description="Convert each panorama image to 6 perspective images")
    parser.add_argument("--matrix_s2t_path", "-m", type=str, required=True, help="Path to the matrix from source world coordinate to target world coordinate")
    parser.add_argument("--source_path", "-s", type=str, required=True, help="Path to the source colmap data folder")
    parser.add_argument("--target_path", "-t", type=str, required=True, help="Path to the target colmap data folder")
    parser.add_argument("--output_path", "-o", type=str, required=True, help="Path to the output colmap data folder")

    args = parser.parse_args(sys.argv[1:])

    if args.matrix_s2t_path.endswith(".npy"):
        matrix_s2t = np.load(args.matrix_s2t_path)
    elif args.matrix_s2t_path.endswith(".txt"):
        matrix_s2t = np.loadtxt(args.matrix_s2t_path, delimiter=',')
    else:
        raise ValueError("Matrix file format not supported")

    merge_data(matrix_s2t, args.source_path, args.target_path, args.output_path)