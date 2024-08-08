#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from typing import NamedTuple
from cam_pose_utils.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, rotmat2qvec, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from cam_pose_utils.graphic_utils import getWorld2View, focal2fov, fov2focal
import numpy as np
import pickle
import math

class CameraInfo(NamedTuple):
    uid: int
    qvec: np.array
    tvec: np.array
    intr_array: np.array
    # FovY: np.array
    # FovX: np.array
    # image: np.array
    # image_path: str
    image_name: str
    # width: int
    # height: int

def radian2angle(radian):
    return radian * 180 / math.pi

def readPklCameras(path):
    pkl_list = os.listdir(path)
    cam_infos = []
    for idx, cam in enumerate(pkl_list):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(pkl_list)))
        sys.stdout.flush()
        with open(os.path.join(path, cam), "rb") as f:
            cam_info_dict = pickle.load(f)

        # H, W
        height = cam_info_dict["image_height"]
        width = cam_info_dict["image_width"]
        
        # extrinsic parameters
        w2c = cam_info_dict["world_view_transform"].transpose()
        c2w = np.linalg.inv(w2c)
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        opencv2blender = np.linalg.inv(blender2opencv)
        c2w = np.matmul(c2w, opencv2blender)
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]
        qvec = rotmat2qvec(R_c2w)
        tvec = T_c2w

        # intrinsic parameters

        FovX = cam_info_dict['FoVx']
        pixel_aspect_x = 1.00
        intr_array = np.array([FovX, pixel_aspect_x, width, height, radian2angle(FovX)])
        cam_info = CameraInfo(uid=idx, qvec=qvec, tvec=tvec, intr_array=intr_array, image_name=cam.split(".")[0])
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readPklSceneInfo(path):
    cam_infos_unsorted = readPklCameras(path=path)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    return cam_infos

def readColmapCameras(cam_extrinsics, cam_intrinsics):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        uid = extr.id

        # H, W
        height = intr.height
        width = intr.width
        
        # extrinsic parameters
        R = qvec2rotmat(extr.qvec)
        T = np.array(extr.tvec)

        w2c = getWorld2View(R, T)
        c2w = np.linalg.inv(w2c)
        blender2opencv = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
        opencv2blender = np.linalg.inv(blender2opencv)
        c2w = np.matmul(c2w, opencv2blender)
        R_c2w = c2w[:3, :3]
        T_c2w = c2w[:3, 3]
        qvec = rotmat2qvec(R_c2w)
        tvec = T_c2w

        # intrinsic parameters
        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            pixel_aspect_x = 1.0

        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
            pixel_aspect_x = focal_length_y / focal_length_x
            # focal_length = focal_length_x * pixel_aspect_x
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"
        
        # intr_array = np.array([focal_length, pixel_aspect_x, width, height])
        intr_array = np.array([FovX, pixel_aspect_x, width, height, radian2angle(FovX)])
        cam_info = CameraInfo(uid=uid, qvec=qvec, tvec=tvec, intr_array=intr_array, image_name=os.path.basename(extr.name).split(".")[0])
        # , image=image, image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readColmapSceneInfo(path):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    # reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    return cam_infos

def save_to_txt(cam_infos, save_dir):
    for cam_info in cam_infos:
        extr_save_dir = os.path.join(save_dir, "extrinsics")
        if not os.path.exists(extr_save_dir):
            os.makedirs(extr_save_dir)
        intr_save_dir = os.path.join(save_dir, "intrinsics")
        if not os.path.exists(intr_save_dir):
            os.makedirs(intr_save_dir)
        np.savetxt(os.path.join(extr_save_dir, cam_info.image_name + "_q.txt"), cam_info.qvec)
        np.savetxt(os.path.join(extr_save_dir, cam_info.image_name + "_t.txt"), cam_info.tvec)
        np.savetxt(os.path.join(intr_save_dir, cam_info.image_name + ".txt"), cam_info.intr_array)