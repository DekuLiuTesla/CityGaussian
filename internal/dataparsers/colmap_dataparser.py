import os
import math
import json
from dataclasses import dataclass
from typing import Optional, Literal, Tuple

import torch
import numpy as np

import internal.utils.colmap as colmap_utils
from internal.cameras.cameras import Cameras
from internal.dataparsers.dataparser import DataParserConfig, DataParser, ImageSet, PointCloud, DataParserOutputs


@dataclass
class Colmap(DataParserConfig):
    """
        Args:
            image_dir: the path to the directory that store images

            mask_dir:
                the path to the directory store mask files;
                the mask file of the image `a/image_name.jpg` is `a/image_name.jpg.png`;
                single channel, 0 is the masked pixel;

            split_mode: reconstruction: train model use all images; experiment: withholding a test set for evaluation

            eval_step: -1: use all images as training set; > 1: pick an image for every eval_step

            reorient: whether reorient the scene

            appearance_groups: filename without extension
    """

    image_dir: str = None

    mask_dir: str = None

    split_mode: Literal["reconstruction", "experiment"] = "reconstruction"

    eval_image_select_mode: Literal["step", "ratio", "list", "list-optional"] = "step"

    eval_step: int = 8

    eval_ratio: float = 0.01

    eval_list: str = None

    scene_scale: float = 1.

    reorient: bool = False  # TODO

    appearance_groups: Optional[str] = None

    image_list: Optional[str] = None

    down_sample_factor: int = 1

    down_sample_rounding_mode: Literal["floor", "round", "round_half_up", "ceil"] = "round"

    points_from: Literal["sfm", "random", "ply"] = "sfm"

    ply_file: str = None

    n_random_points: int = 100_000

    force_pinhole: bool = False

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return ColmapDataParser(path, output_path, global_rank, self)


class ColmapDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: Colmap) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def _round_half_up(self, i: torch.Tensor):
        return torch.floor(i + 0.5)

    def detect_sparse_model_dir(self) -> str:
        if os.path.isdir(os.path.join(self.path, "sparse", "0")):
            return os.path.join(self.path, "sparse", "0")
        return os.path.join(self.path, "sparse")

    def get_image_dir(self) -> str:
        if self.params.image_dir is None:
            image_dir = os.path.join(self.path, "images")
            if self.params.down_sample_factor > 1:
                image_dir = image_dir + "_{}".format(self.params.down_sample_factor)
            return image_dir
        return os.path.join(self.path, self.params.image_dir)

    @staticmethod
    def rotation_matrix(a, b):
        """Compute the rotation matrix that rotates vector a to vector b.

        Args:
            a: The vector to rotate.
            b: The vector to rotate to.
        Returns:
            The rotation matrix.
        """
        a = a / torch.linalg.norm(a)
        b = b / torch.linalg.norm(b)
        v = torch.cross(a, b, dim=-1)
        c = torch.dot(a, b)
        # If vectors are exactly opposite, we add a little noise to one of them
        if c < -1 + 1e-8:
            eps = (torch.rand(3, dtype=a.dtype, device=a.device) - 0.5) * 0.01
            return ColmapDataParser.rotation_matrix(a + eps, b)
        s = torch.linalg.norm(v)
        skew_sym_mat = torch.tensor(
            [
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0],
            ],
            dtype=a.dtype,
            device=a.device,
        )
        return torch.eye(3, dtype=a.dtype, device=a.device) + skew_sym_mat + skew_sym_mat @ skew_sym_mat * ((1 - c) / (s ** 2 + 1e-8))

    @staticmethod
    def read_points3D_binary(path_to_model_file, selected_image_ids: dict = None):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """

        with open(path_to_model_file, "rb") as fid:
            num_points = colmap_utils.read_next_bytes(fid, 8, "Q")[0]

            xyzs = []
            rgbs = []
            errors = []

            for p_id in range(num_points):
                binary_point_line_properties = colmap_utils.read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = colmap_utils.read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = colmap_utils.read_next_bytes(
                    fid, num_bytes=8 * track_length,
                    format_char_sequence="ii" * track_length)

                # whether point belongs to selected images
                if selected_image_ids is not None:
                    image_ids = np.array(tuple(map(int, track_elems[0::2])))
                    point_in_selected_image_count = 0
                    for image_id in image_ids:
                        if image_id in selected_image_ids:
                            point_in_selected_image_count += 1
                    if point_in_selected_image_count == 0:
                        continue

                # TODO: filter points in masked area

                xyzs.append(xyz)
                rgbs.append(rgb)
                errors.append(error)
        return np.asarray(xyzs), np.asarray(rgbs), np.asarray(errors)

    def get_outputs(self) -> DataParserOutputs:
        # load colmap sparse model
        sparse_model_dir = self.detect_sparse_model_dir()
        cameras = colmap_utils.read_cameras_binary(os.path.join(sparse_model_dir, "cameras.bin"))
        images = colmap_utils.read_images_binary(os.path.join(sparse_model_dir, "images.bin"))

        # sort images
        images = dict(sorted(images.items(), key=lambda item: item[0]))

        # filter images
        selected_image_ids = None
        selected_image_names = None
        if self.params.image_list is not None:
            # load image list
            selected_image_ids = {}
            selected_image_names = {}
            with open(self.params.image_list, "r") as f:
                for image_name in f:
                    image_name = image_name[:-1]
                    selected_image_names[image_name] = True
            # filter images by image list
            new_images = {}
            for i in images:
                image = images[i]
                if image.name in selected_image_names:
                    selected_image_ids[image.id] = True
                    new_images[i] = image
            assert len(new_images) > 0, "no image left after filtering via {}".format(self.params.image_list)

            # replace images with new_images
            images = new_images

        image_dir = self.get_image_dir()

        # build appearance dict: group name -> image name list
        if self.params.appearance_groups is None:
            print("appearance group by camera id")
            appearance_groups = {}
            for i in images:
                image_camera_id = images[i].camera_id
                if image_camera_id not in appearance_groups:
                    appearance_groups[image_camera_id] = []
                appearance_groups[image_camera_id].append(images[i].name)
        else:
            appearance_group_file_path = os.path.join(self.path, self.params.appearance_groups)
            print("loading appearance groups from {}".format(appearance_group_file_path))
            with open("{}.json".format(appearance_group_file_path), "r") as f:
                appearance_groups = json.load(f)

        # sort the appearance group name list
        appearance_group_name_list = sorted(list(appearance_groups.keys()))
        appearance_group_num = float(len(appearance_group_name_list))

        # use order as the appearance id of the group
        # map from image name to appearance id
        image_name_to_appearance_id = {}
        image_name_to_normalized_appearance_id = {}
        appearance_group_name_to_appearance_id = {}  # tuple(id, normalized id)
        for idx, appearance_group_name in enumerate(appearance_group_name_list):
            normalized_idx = idx / appearance_group_num
            appearance_group_name_to_appearance_id[appearance_group_name] = (idx, normalized_idx)
            for image_name in appearance_groups[appearance_group_name]:
                image_name_to_appearance_id[image_name] = idx
                image_name_to_normalized_appearance_id[image_name] = normalized_idx

        # convert appearance id dict to list, which use the same order as the colmap images
        image_appearance_id = []
        image_normalized_appearance_id = []
        for i in images:
            image_appearance_id.append(image_name_to_appearance_id[images[i].name])
            image_normalized_appearance_id.append(image_name_to_normalized_appearance_id[images[i].name])

        # convert points3D to ply
        # ply_path = os.path.join(sparse_model_dir, "points3D.ply")
        # while os.path.exists(ply_path) is False:
        #     if self.global_rank == 0:
        #         print("converting points3D.bin to ply format")
        #         xyz, rgb, _ = ColmapDataParser.read_points3D_binary(os.path.join(sparse_model_dir, "points3D.bin"))
        #         ColmapDataParser.convert_points_to_ply(ply_path + ".tmp", xyz=xyz, rgb=rgb)
        #         os.rename(ply_path + ".tmp", ply_path)
        #         break
        #     else:
        #         # waiting ply
        #         print("#{} waiting for {}".format(os.getpid(), ply_path))
        #         time.sleep(1)
        if self.params.points_from == "sfm":
            print("loading colmap 3D points")
            xyz, rgb, _ = ColmapDataParser.read_points3D_binary(
                os.path.join(sparse_model_dir, "points3D.bin"),
                selected_image_ids=selected_image_ids,
            )
        else:
            # random points generated later
            xyz = np.ones((1, 3))
            rgb = np.ones((1, 3))

        loaded_mask_count = 0
        # initialize lists
        R_list = []
        T_list = []
        fx_list = []
        fy_list = []
        # fov_x_list = []
        # fov_y_list = []
        cx_list = []
        cy_list = []
        width_list = []
        height_list = []
        appearance_id_list = image_appearance_id
        normalized_appearance_id_list = image_normalized_appearance_id
        camera_type_list = []
        image_name_list = []
        image_path_list = []
        mask_path_list = []

        # parse colmap sparse model
        for idx, key in enumerate(images):
            # extract image and its correspond camera
            extrinsics = images[key]
            intrinsics = cameras[extrinsics.camera_id]

            height = intrinsics.height
            width = intrinsics.width

            R = extrinsics.qvec2rotmat()
            T = np.array(extrinsics.tvec)

            if intrinsics.model == "SIMPLE_PINHOLE":
                focal_length_x = intrinsics.params[0]
                focal_length_y = focal_length_x
                cx = intrinsics.params[1]
                cy = intrinsics.params[2]
                # fov_y = focal2fov(focal_length_x, height)
                # fov_x = focal2fov(focal_length_x, width)
            elif intrinsics.model == "PINHOLE" or self.params.force_pinhole is True:
                focal_length_x = intrinsics.params[0]
                focal_length_y = intrinsics.params[1]
                cx = intrinsics.params[2]
                cy = intrinsics.params[3]
                # fov_y = focal2fov(focal_length_y, height)
                # fov_x = focal2fov(focal_length_x, width)
            else:
                undistorted_output_dir = os.path.join(self.path, "dense")
                raise RuntimeError("Unsupported camera model: only PINHOLE or SIMPLE_PINHOLE currently. Please undistort your images with the command below first:\n  colmap image_undistorter --image_path {} --input_path {} --output_path {}\nthen use `{}` as the value of `--data.path`.".format(image_dir, sparse_model_dir, undistorted_output_dir, undistorted_output_dir))

            # whether mask exists
            mask_path = None
            if self.params.mask_dir is not None:
                mask_path = os.path.join(self.params.mask_dir, "{}.png".format(extrinsics.name))
                if os.path.exists(mask_path) is True:
                    loaded_mask_count += 1
                else:
                    mask_path = None

            # append data to list
            R_list.append(R)
            T_list.append(T)
            fx_list.append(focal_length_x)
            fy_list.append(focal_length_y)
            # fov_x_list.append(fov_x)
            # fov_y_list.append(fov_y)
            cx_list.append(cx)
            cy_list.append(cy)
            width_list.append(width)
            height_list.append(height)
            camera_type_list.append(0)
            image_name_list.append(extrinsics.name)
            image_path_list.append(os.path.join(image_dir, extrinsics.name))
            mask_path_list.append(mask_path)

        # loaded mask must not be zero if self.params.mask_dir provided
        if self.params.mask_dir is not None and loaded_mask_count == 0:
            raise RuntimeError("not a mask was loaded from {}, "
                               "please remove the mask_dir parameter if this is a expected result".format(
                self.params.mask_dir
            ))

        # calculate norm
        # norm = getNerfppNorm(R_list, T_list)

        # convert data to tensor
        R = torch.tensor(np.stack(R_list, axis=0), dtype=torch.float32)
        T = torch.tensor(np.stack(T_list, axis=0), dtype=torch.float32)
        fx = torch.tensor(fx_list, dtype=torch.float32)
        fy = torch.tensor(fy_list, dtype=torch.float32)
        # fov_x = torch.tensor(fov_x_list, dtype=torch.float32)
        # fov_y = torch.tensor(fov_y_list, dtype=torch.float32)
        cx = torch.tensor(cx_list, dtype=torch.float32)
        cy = torch.tensor(cy_list, dtype=torch.float32)
        width = torch.tensor(width_list, dtype=torch.int16)
        height = torch.tensor(height_list, dtype=torch.int16)
        appearance_id = torch.tensor(appearance_id_list, dtype=torch.int)
        normalized_appearance_id = torch.tensor(normalized_appearance_id_list, dtype=torch.float32)
        camera_type = torch.tensor(camera_type_list, dtype=torch.int8)

        # recalculate intrinsics if down sample enabled
        if self.params.down_sample_factor != 1:
            if self.params.down_sample_rounding_mode == "round_half_up":
                rounding_func = self._round_half_up
            else:
                rounding_func = getattr(torch, self.params.down_sample_rounding_mode)
            down_sampled_width = rounding_func(width.to(torch.float) / self.params.down_sample_factor)
            down_sampled_height = rounding_func(height.to(torch.float) / self.params.down_sample_factor)
            width_scale_factor = down_sampled_width / width
            height_scale_factor = down_sampled_height / height
            fx *= width_scale_factor
            fy *= height_scale_factor
            cx *= width_scale_factor
            cy *= height_scale_factor

            width = down_sampled_width.to(torch.int16)
            height = down_sampled_height.to(torch.int16)

            print("down sample enabled")

        is_w2c_required = self.params.scene_scale != 1.0 or self.params.reorient is True
        if is_w2c_required:
            # build world-to-camera transform matrix
            w2c = torch.zeros(size=(R.shape[0], 4, 4))
            w2c[:, :3, :3] = R
            w2c[:, :3, 3] = T
            w2c[:, 3, 3] = 1.
            # convert to camera-to-world transform matrix
            c2w = torch.linalg.inv(w2c)

            # reorient
            if self.params.reorient is True:
                print("reorient scene")

                # calculate rotation transform matrix
                up = -torch.mean(c2w[:, :3, 1], dim=0)
                up = up / torch.linalg.norm(up)

                print("up vector = {}".format(up.numpy()))

                rotation = self.rotation_matrix(up, torch.Tensor([0, 0, 1]))
                rotation_transform = torch.eye(4)
                rotation_transform[:3, :3] = rotation

                # reorient cameras
                print("reorienting cameras...")
                c2w = torch.matmul(rotation_transform, c2w)
                # reorient points
                print("reorienting points...")
                xyz = np.matmul(xyz, rotation.numpy().T)

            # rescale scene size
            if self.params.scene_scale != 1.0:
                print("rescal scene with factor {}".format(self.params.scene_scale))

                # rescale camera poses
                c2w[:, :3, 3] *= self.params.scene_scale
                # rescale point cloud
                xyz *= self.params.scene_scale

                # rescale scene extent
                # norm["radius"] *= self.params.scene_scale

            # convert back to world-to-camera
            w2c = torch.linalg.inv(c2w)
            R = w2c[:, :3, :3]
            T = w2c[:, :3, 3]

        # build split indices
        training_set_indices, validation_set_indices = self.build_split_indices(image_name_list)

        # split
        image_set = []
        for index_list in [training_set_indices, validation_set_indices]:
            indices = torch.tensor(index_list, dtype=torch.int)
            cameras = Cameras(
                R=R[indices],
                T=T[indices],
                fx=fx[indices],
                fy=fy[indices],
                cx=cx[indices],
                cy=cy[indices],
                width=width[indices],
                height=height[indices],
                appearance_id=appearance_id[indices],
                normalized_appearance_id=normalized_appearance_id[indices],
                distortion_params=None,
                camera_type=camera_type[indices],
            )
            image_set.append(ImageSet(
                image_names=[image_name_list[i] for i in index_list],
                image_paths=[image_path_list[i] for i in index_list],
                mask_paths=[mask_path_list[i] for i in index_list],
                cameras=cameras
            ))

        if self.params.points_from == "random":
            print("generate {} random points".format(self.params.n_random_points))
            scene_center = torch.mean(image_set[0].cameras.camera_center, dim=0)
            scene_radius = (image_set[0].cameras.camera_center - scene_center).norm(dim=-1).max()
            xyz = (np.random.random((self.params.n_random_points, 3)) * 2. - 1.) * 3 * scene_radius.numpy() + scene_center.numpy()
            rgb = (np.random.random((self.params.n_random_points, 3)) * 255).astype(np.uint8)
        elif self.params.points_from == "ply":
            assert self.params.ply_file is not None
            from internal.utils.graphics_utils import fetch_ply_without_rgb_normalization
            basic_pcd = fetch_ply_without_rgb_normalization(os.path.join(self.path, self.params.ply_file))
            xyz = basic_pcd.points
            rgb = basic_pcd.colors
            print("load {} points from {}".format(xyz.shape[0], self.params.ply_file))

        # print information
        print("[colmap dataparser] train set images: {}, val set images: {}, loaded mask: {}".format(
            len(image_set[0]),
            len(image_set[1]),
            loaded_mask_count,
        ))

        return DataParserOutputs(
            train_set=image_set[0],
            val_set=image_set[1],
            test_set=image_set[1],
            point_cloud=PointCloud(
                xyz=xyz,
                rgb=rgb,
            ),
            # camera_extent=norm["radius"],
            appearance_group_ids=appearance_group_name_to_appearance_id,
        )

    def build_eval_list_split_indices(self, image_name_list) -> Tuple[list, list]:
        assert self.params.eval_list is not None

        eval_image_set = {}
        with open(self.params.eval_list, "r") as f:
            for row in f:
                row = row.rstrip("\n")
                eval_image_set[row] = True

        train_set_indices = []
        eval_set_indices = []

        for idx, name in enumerate(image_name_list):
            if name in eval_image_set:
                eval_set_indices.append(idx)
                del eval_image_set[name]
                if self.params.split_mode == "experiment":
                    continue
            train_set_indices.append(idx)

        if len(eval_image_set) != 0:
            message = "Some images can not be found in colmap sparse model: {}".format(list(eval_image_set.keys()))
            if self.params.split_mode == "list":
                raise RuntimeError(message)
            print("[WARNING]{}".format(message))

        if len(eval_set_indices) == 0:
            eval_set_indices = train_set_indices[:1]

        return train_set_indices, eval_set_indices

    def build_step_split_indices(self, image_name_list) -> Tuple[list, list]:
        assert self.params.eval_step > 1, "eval_step must > 1"
        eval_step = self.params.eval_step
        if self.params.eval_image_select_mode == "ratio":
            eval_image_num = max(math.ceil(self.params.eval_ratio * len(image_name_list)), 1)
            eval_step = len(image_name_list) // eval_image_num

        if self.params.split_mode == "experiment":
            # split train set and val set
            training_set_indices = []
            validation_set_indices = []
            for i in range(len(image_name_list)):
                if i % eval_step == 0:
                    validation_set_indices.append(i)
                else:
                    training_set_indices.append(i)
        else:
            # train set contains val set
            training_set_indices = list(range(len(image_name_list)))
            validation_set_indices = training_set_indices[::eval_step]

        return training_set_indices, validation_set_indices

    def build_split_indices(self, image_name_list) -> Tuple[list, list]:
        if self.params.eval_image_select_mode.startswith("list"):
            return self.build_eval_list_split_indices(image_name_list)
        assert self.params.eval_list is None, "eval_image_select_mode=='{}', but eval_list is not None".format(self.params.eval_image_select_mode)
        return self.build_step_split_indices(image_name_list)
