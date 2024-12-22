import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__)))))

import cv2
import json
import xml.etree.ElementTree as ET
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
from plyfile import PlyData, PlyElement
from internal.utils.las_utils import read_las_fit

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Distort raw aerial images and save extrinsics and intrinsics")
    parser.add_argument("--aerial_dir_path", type=str, default="data/GKD/aerial")
    parser.add_argument("--aerial_xml_file", type=str, default="data/GKD/aerial/info.xml")
    parser.add_argument("--convert_pcd", action="store_true")
    parser.add_argument("--pcd_ds_interval", type=int, default=100)
    parser.add_argument("--skip_img_saving", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    tree = ET.parse(args.aerial_xml_file)
    root = tree.getroot()

    num_images = len(root.findall(".//Photogroup/Photo"))

    f = float(root.find(f".//Photogroup/FocalLengthPixels").text)
    cx = float(root.find(f".//Photogroup/PrincipalPoint/x").text)
    cy = float(root.find(f".//Photogroup/PrincipalPoint/y").text)
    width = int(root.find(f".//Photogroup/ImageDimensions/Width").text)
    height = int(root.find(f".//Photogroup/ImageDimensions/Height").text)

    K1 = float(root.find(f".//Photogroup/Distortion/K1").text)
    K2 = float(root.find(f".//Photogroup/Distortion/K2").text)
    K3 = float(root.find(f".//Photogroup/Distortion/K3").text)
    P1 = float(root.find(f".//Photogroup/Distortion/P1").text)
    P2 = float(root.find(f".//Photogroup/Distortion/P2").text)

    if args.convert_pcd:
        print(f"Converting aerial point cloud to .ply file, downsample interval: {args.pcd_ds_interval}")
        aerial_pcd_path = os.path.join(args.aerial_dir_path, 'pcd')
        aerial_pcd_list = sorted([os.path.join(aerial_pcd_path, f) for f in os.listdir(aerial_pcd_path) if f.endswith('.las')])
        xyz_aerial_list, rgb_aerial_list = [], []
        for aerial_pcd_file in aerial_pcd_list:
            xyz_aerial, rgb_aerial, _ = read_las_fit(aerial_pcd_file, ["scales", "offsets"])
            xyz_aerial_list.append(xyz_aerial[::args.pcd_ds_interval])
            rgb_aerial_list.append(rgb_aerial[::args.pcd_ds_interval])

        xyz_aerial = np.concatenate(xyz_aerial_list, axis=0)
        rgb_aerial = np.concatenate(rgb_aerial_list, axis=0)

        # save as .ply file, note that rgb is (0, 1) float
        rgb_aerial = (rgb_aerial * 255).astype(np.uint8)
        dtype_full = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
        attributes = np.concatenate([xyz_aerial, rgb_aerial], axis=1)
        elements = np.empty(xyz_aerial.shape[0], dtype=dtype_full)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(os.path.join(args.aerial_dir_path, 'point_cloud.ply'))
        print(f"Saved {xyz_aerial.shape[0]} point cloud to {os.path.join(args.aerial_dir_path, 'point_cloud.ply')}")
    
    transform_path = os.path.join(args.aerial_dir_path, 'transforms.json')
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

    for cam_idx in tqdm(range(0, num_images)):
        M_00 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_00").text)
        M_01 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_01").text)
        M_02 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_02").text)
        M_10 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_10").text)
        M_11 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_11").text)
        M_12 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_12").text)
        M_20 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_20").text)
        M_21 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_21").text)
        M_22 = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Rotation/M_22").text)

        X = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Center/x").text)
        Y = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Center/y").text)
        Z = float(root.find(f".//Photogroup/Photo[Id='{cam_idx}']/Pose/Center/z").text)

        xml_image_path = root.find(f".//Photogroup/Photo[Id='{cam_idx}']/ImagePath").text
        aerial_image_path = os.path.join(args.aerial_dir_path, xml_image_path.split('/')[-2], xml_image_path.split('/')[-1])
        aerial_image = cv2.imread(aerial_image_path)
        aerial_image = cv2.cvtColor(aerial_image, cv2.COLOR_BGR2RGB)
        H, W = aerial_image.shape[:2]

        c2w = np.array([[M_00, M_10, M_20, X], 
                        [M_01, M_11, M_21, Y], 
                        [M_02, M_12, M_22, Z], 
                        [0, 0, 0, 1]])
        w2c = np.linalg.inv(c2w)
        intrinsics = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])

        new_intrinsics, roi = cv2.getOptimalNewCameraMatrix(intrinsics, np.array([K1, K2, P1, P2, K3]), (W, H), 1, (W, H))
        undistorted_image = cv2.undistort(aerial_image, intrinsics, np.array([K1, K2, P1, P2, K3]), newCameraMatrix=new_intrinsics)
        x, y, w, h = roi
        undistorted_image = undistorted_image[y:y+h, x:x+w]
        new_intrinsics[0, 2] -= x
        new_intrinsics[1, 2] -= y
        undistorted_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)

        if not args.skip_img_saving:
            os.makedirs(os.path.join(args.aerial_dir_path, 'images'), exist_ok=True)
            cv2.imwrite(os.path.join(args.aerial_dir_path, 'images', f"{cam_idx:05d}.png"), undistorted_image)

        transform_dict['frames'].append({
            "frame_index": cam_idx,
            "file_path": os.path.join(xml_image_path.split('/')[-2], xml_image_path.split('/')[-1]),
            "rot_mat": c2w.tolist()
        })
    
    # add camera intrinsics
    transform_dict["camera_angle_x"] = np.arctan(new_intrinsics[0, 2] / new_intrinsics[0, 0]) * 2
    transform_dict["fl_x"] = new_intrinsics[0, 0]
    transform_dict["fl_y"] = new_intrinsics[1, 1]
    transform_dict["cx"] = new_intrinsics[0, 2]
    transform_dict["cy"] = new_intrinsics[1, 2]
    transform_dict["w"] = w
    transform_dict["h"] = h

    with open(transform_path, 'w') as f:
        json.dump(transform_dict, f, indent=4)
        print(f"Saved {num_images} aerial images' transforms to {transform_path}")
    
