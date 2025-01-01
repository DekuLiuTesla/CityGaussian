from typing import Literal
import os
import json

import numpy as np
import torch
from dataclasses import dataclass
from internal.cameras.cameras import CameraType
from .dataparser import DataParserConfig, DataParser, ImageSet, Cameras, PointCloud, DataParserOutputs


@dataclass
class NGP(DataParserConfig):
    name: str = "transforms.json"

    pcd_from: Literal["auto", "random", "file"] = "auto"

    pcd_file: str = "points3D.ply"

    num_random_points: int = 100_000

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return NGPDataParser(self, path)


class NGPDataParser(DataParser):
    def __init__(self, config, path):
        super().__init__()

        self.config = config
        self.path = path

    def get_outputs(self) -> DataParserOutputs:
        with open(os.path.join(self.path, self.config.name), "r") as f:
            transforms = json.load(f)

        val_set_filename_list = transforms.get("val_filenames", [])
        test_set_filename_list = transforms.get("test_filenames", [])

        train_set_index_list = []
        val_set_index_list = []
        test_set_index_list = []

        file_name_list = []
        file_path_list = []
        depth_path_list = []
        mask_path_list = []
        fx_list = []
        fy_list = []
        cx_list = []
        cy_list = []
        width_list = []
        height_list = []
        camera_type_list = []
        distortion_params_list = []
        c2w_list = []

        # read frames
        for idx, frame in enumerate(transforms["frames"]):
            if frame["file_path"] in val_set_filename_list:
                val_set_index_list.append(idx)
            elif frame["file_path"] in test_set_filename_list:
                test_set_index_list.append(idx)
            else:
                train_set_index_list.append(idx)

            file_name_list.append(frame["file_path"])
            file_path_list.append(os.path.join(self.path, frame["file_path"]))

            if "depth_file_path" in frame:
                depth_path_list.append(os.path.join(self.path, frame["depth_file_path"]))
            else:
                depth_path_list.append(None)

            if "mask_path" in frame:
                mask_path_list.append(os.path.join(self.path, frame["mask_path"]))
            else:
                mask_path_list.append(None)

            c2w_list.append(frame["transform_matrix"])

            # camera intrinsics
            if "fl_x" in frame:
                # image specified camera intrinsics
                intrinsics_dict = frame
            else:
                intrinsics_dict = transforms

            fx_list.append(intrinsics_dict["fl_x"])
            fy_list.append(intrinsics_dict["fl_y"])
            cx_list.append(intrinsics_dict["cx"])
            cy_list.append(intrinsics_dict["cy"])
            width_list.append(intrinsics_dict["w"])
            height_list.append(intrinsics_dict["h"])

            camera_model = frame.get("camera_model", transforms.get("camera_model", None))
            if camera_model is None or camera_model == "OPENCV":
                camera_type_list.append(CameraType.PERSPECTIVE)
            elif camera_model == "OPENCV_FISHEYE":
                camera_type_list.append(CameraType.FISHEYE)
            else:
                raise ValueError("Unsupported camera_model '{}' of image '{}'".format(camera_model, frame["file_path"]))

            distortion_params = []
            for param_name in ["k1", "k2", "p1", "p2", "k3", "k4"]:
                distortion_params.append(intrinsics_dict.get(param_name, 0.))
            distortion_params_list.append(distortion_params)

        # convert lists to tensors
        fx = torch.tensor(fx_list, dtype=torch.float)
        fy = torch.tensor(fy_list, dtype=torch.float)
        cx = torch.tensor(cx_list, dtype=torch.float)
        cy = torch.tensor(cy_list, dtype=torch.float)
        width = torch.tensor(width_list, dtype=torch.int)
        height = torch.tensor(height_list, dtype=torch.int)
        camera_types = torch.tensor(camera_type_list, dtype=torch.int)
        distortion_params = torch.tensor(distortion_params_list, dtype=torch.float)
        c2w = torch.tensor(c2w_list, dtype=torch.float64)
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:, :3, 1:3] *= -1
        w2c = torch.linalg.inv(c2w).to(torch.float)
        R = w2c[:, :3, :3]
        T = w2c[:, :3, 3]

        appearance_id = torch.arange(0, fx.shape[0], dtype=torch.int)
        normalized_appearance_id = appearance_id.to(torch.float32) / appearance_id[-1]

        def generate_image_set(indices: list):
            tensor_indices = torch.tensor(indices, dtype=torch.int)
            return ImageSet(
                image_names=[file_name_list[i] for i in indices],
                image_paths=[file_path_list[i] for i in indices],
                cameras=Cameras(
                    R=R[tensor_indices],
                    T=T[tensor_indices],
                    fx=fx[tensor_indices],
                    fy=fy[tensor_indices],
                    cx=cx[tensor_indices],
                    cy=cy[tensor_indices],
                    width=width[tensor_indices],
                    height=height[tensor_indices],
                    appearance_id=appearance_id[tensor_indices],
                    normalized_appearance_id=normalized_appearance_id[tensor_indices],
                    distortion_params=distortion_params[tensor_indices],
                    camera_type=camera_types[tensor_indices],
                ),
                depth_paths=[depth_path_list[i] for i in indices],
                mask_paths=[mask_path_list[i] for i in indices],
            )

        assert len(train_set_index_list) > 0

        # point cloud
        pcd_file_path = os.path.join(self.path, self.config.pcd_file)
        is_pcd_file_exist = os.path.exists(pcd_file_path)
        if self.config.pcd_from == "file" and is_pcd_file_exist is False:
            raise ValueError(f"'{pcd_file_path}' not exists")

        if self.config.pcd_from == "file" or self.config.pcd_from == "auto" and is_pcd_file_exist is True:
            print(f"Load pcd from '{pcd_file_path}'")
            from internal.utils.graphics_utils import fetch_ply
            pcd = fetch_ply(pcd_file_path)
            point_cloud = PointCloud(
                xyz=pcd.points,
                rgb=(pcd.colors * 255).astype(np.uint8),
            )
        else:
            print("Generate random pcd")
            camera_centers = c2w[:, :3, 3]
            scene_center = camera_centers.mean(dim=0)
            max_radius = (camera_centers - scene_center).norm(dim=-1).max().item()

            xyz = (np.random.random((self.config.num_random_points, 3)) * 2 - 1.) * max_radius * 1.5 + scene_center.numpy()
            rgb = (np.random.random((self.config.num_random_points, 3)) * 255).astype(np.uint8)

            point_cloud = PointCloud(
                xyz=xyz,
                rgb=rgb,
            )

        return DataParserOutputs(
            train_set=generate_image_set(train_set_index_list),
            val_set=generate_image_set(val_set_index_list if len(val_set_index_list) > 0 else train_set_index_list[:1]),
            test_set=generate_image_set(test_set_index_list if len(test_set_index_list) > 0 else train_set_index_list[:1]),
            point_cloud=point_cloud,
            appearance_group_ids={filename: (appearance_id[idx].item(), normalized_appearance_id[idx].item()) for idx, filename in enumerate(file_name_list)},
        )
