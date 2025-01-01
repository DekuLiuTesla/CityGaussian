import os
import math
import json
import numpy as np
import torch

from typing import Tuple
from PIL import Image
from tqdm.auto import tqdm
from dataclasses import dataclass
from internal.cameras.cameras import Cameras
from internal.utils.depth_map_utils import depth_map_to_colored_points, read_depth_as_tensor
from .dataparser import ImageSet, PointCloud, DataParserConfig, DataParser, DataParserOutputs


@dataclass
class MatrixCity(DataParserConfig):
    train: list[str] = None

    test: list[str] = None

    scale: float = 0.01
    """ Scene size scale """

    depth_scale: float = 0.01
    """ Do not change this """

    max_depth: float = 65_000
    """ Using to remove sky, multiply with scale and depth_scale automatically """

    depth_read_step: int = 1
    """ Take every `depth_read_step`th depth map for generating the point cloud """

    max_points: int = 3_840_000
    """ How many points will be used for initialization """

    use_depth: bool = False
    """ Whether load depth maps into training batches. Enable this if you need depth regularization. """

    use_inverse_depth: bool = True

    point_cloud_generation_device: str = "cpu"

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return MatrixCityDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class MatrixCityDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: MatrixCity) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def _parse_json(self, paths, build_point_cloud: bool = False) -> Tuple[ImageSet, PointCloud]:
        image_names = []
        image_paths = []
        depth_paths = []
        c2w_tensor_list = []
        R_tensor_list = []
        T_tensor_list = []
        fx_tensor_list = []
        fy_tensor_list = []
        cx_tensor_list = []
        cy_tensor_list = []
        width_tensor_list = []
        height_tensor_list = []
        for json_relative_path in paths:
            path = os.path.join(self.path, json_relative_path)
            with open(path, "r") as f:
                transforms = json.load(f)

            fov_x = transforms["camera_angle_x"]

            # get image shape, assert all image use same shape
            base_dir = os.path.dirname(path)
            if "path" in transforms["frames"][0]:
                base_dir = os.path.join(base_dir, transforms["frames"][0]["path"])
            image = Image.open(os.path.join(
                base_dir,
                "rgb",
                "{:04d}.png".format(0),
            ))
            width = image.width
            height = image.height
            image.close()

            # build image name list and camera poses
            c2w_list = []
            for frame in transforms["frames"]:
                # TODO: load fov provided by frame
                frame_id = frame["frame_index"]
                base_dir = os.path.dirname(path)
                if "path" in frame:
                    base_dir = os.path.join(base_dir, frame["path"])
                image_paths.append(os.path.join(
                    base_dir,
                    "rgb",
                    "{:04d}.png".format(frame_id),
                ))
                depth_paths.append(os.path.join(
                    base_dir,
                    "depth",
                    "{:04d}.exr".format(frame_id),
                ))

                image_names.append("{}/{:04d}".format(os.path.basename(base_dir), frame_id))

                c2w = torch.tensor(frame['rot_mat'], dtype=torch.float64)
                c2w_list.append(c2w)

            # convert c2w to w2c
            camera_to_world = torch.stack(c2w_list)
            camera_to_world[:, :3, :3] *= 100
            camera_to_world[:, :3, 3] *= self.params.scale
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            camera_to_world[:, :3, 1:3] *= -1
            c2w_tensor_list.append(camera_to_world)
            world_to_camera = torch.linalg.inv(camera_to_world).to(torch.float)

            # extract R and T from w2c
            R = world_to_camera[:, :3, :3]
            T = world_to_camera[:, :3, 3]

            # calculate camera intrinsics
            fx = torch.tensor([.5 * width / np.tan(.5 * fov_x)]).expand(R.shape[0])
            fy = fx
            cx = torch.tensor([width / 2]).expand(R.shape[0])
            cy = torch.tensor([height / 2]).expand(R.shape[0])
            width = torch.tensor([width], dtype=torch.float).expand(R.shape[0])
            height = torch.tensor([height], dtype=torch.float).expand(R.shape[0])

            # append to list
            R_tensor_list.append(R)
            T_tensor_list.append(T)
            fx_tensor_list.append(fx)
            fy_tensor_list.append(fy)
            cx_tensor_list.append(cx)
            cy_tensor_list.append(cy)
            width_tensor_list.append(width)
            height_tensor_list.append(height)

        width = torch.concat(width_tensor_list, dim=0)
        cameras = Cameras(
            R=torch.concat(R_tensor_list, dim=0),
            T=torch.concat(T_tensor_list, dim=0),
            fx=torch.concat(fx_tensor_list, dim=0),
            fy=torch.concat(fy_tensor_list, dim=0),
            cx=torch.concat(cx_tensor_list, dim=0),
            cy=torch.concat(cy_tensor_list, dim=0),
            width=width,
            height=torch.concat(height_tensor_list, dim=0),
            appearance_id=torch.zeros_like(width, dtype=torch.int),
            normalized_appearance_id=torch.zeros_like(width),
            distortion_params=None,
            camera_type=torch.zeros_like(width),
        )
        if build_point_cloud is True:
            from internal.utils.graphics_utils import store_ply, fetch_ply_without_rgb_normalization
            import hashlib
            import dataclasses

            # check whether need to regenerate point cloud based on params
            params_dict = dataclasses.asdict(self.params)
            params_dict["train"] = sorted(params_dict["train"])
            del params_dict["test"]  # ignore test set
            del params_dict["use_depth"]
            del params_dict["use_inverse_depth"]
            del params_dict["point_cloud_generation_device"]
            params_json = json.dumps(params_dict, indent=4, ensure_ascii=False)
            print(params_json)
            ply_file_path = os.path.join(
                self.path,
                "{}.ply".format(hashlib.sha1(params_json.encode("utf-8")).hexdigest()),
            )
            if os.path.exists(ply_file_path):
                final_pcd = fetch_ply_without_rgb_normalization(ply_file_path)
                point_cloud = PointCloud(
                    final_pcd.points,
                    final_pcd.colors,
                )
            else:
                c2w = torch.concat(c2w_tensor_list, dim=0).to(dtype=torch.float, device=torch.device(self.params.point_cloud_generation_device))
                torch.inverse(torch.ones((1, 1), device=c2w.device))  # https://github.com/pytorch/pytorch/issues/90613#issuecomment-1817307008

                os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
                import cv2

                points_per_image = math.ceil(self.params.max_points / (len(image_paths) // self.params.depth_read_step))
                final_depth_scale = self.params.scale * self.params.depth_scale

                def read_rgb(path: str):
                    image = cv2.imread(path)
                    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                def build_single_image_3d_points(frame_idx):
                    fx = cameras.fx[frame_idx]
                    fy = cameras.fy[frame_idx]
                    cx = cameras.cx[frame_idx]
                    cy = cameras.cy[frame_idx]

                    # read rgb and depth
                    rgb = read_rgb(image_paths[frame_idx])
                    depth = read_depth_as_tensor(depth_paths[frame_idx]).to(dtype=c2w.dtype, device=c2w.device) * final_depth_scale

                    valid_pixel_mask = depth < self.params.max_depth * final_depth_scale

                    points_3d_in_world, rgb = depth_map_to_colored_points(
                        depth_map=depth,
                        rgb=rgb,
                        fx=fx,
                        fy=fy,
                        cx=cx,
                        cy=cy,
                        c2w=c2w[frame_idx],
                        valid_pixel_mask=valid_pixel_mask,
                    )

                    # random sample
                    n_points = points_3d_in_world.shape[0]
                    if points_per_image < n_points:
                        sample_indices = torch.randperm(n_points)[:points_per_image]
                        points_3d_in_world = points_3d_in_world[sample_indices]
                        rgb = rgb[sample_indices.cpu().numpy()]

                    return points_3d_in_world.cpu(), rgb

                xyz_list = []
                rgb_list = []

                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=8) as tpe:
                    read_depth_frame_id_list = list(range(len(image_paths)))[::self.params.depth_read_step]
                    for xyz, rgb in tqdm(tpe.map(
                            build_single_image_3d_points,
                            read_depth_frame_id_list,
                    ), total=len(read_depth_frame_id_list)):
                        xyz_list.append(xyz)
                        rgb_list.append(rgb)

                point_cloud = PointCloud(
                    torch.concat(xyz_list, dim=0).numpy(),
                    np.concatenate(rgb_list, axis=0),
                )

                if self.params.point_cloud_generation_device != "cpu":
                    del c2w
                    torch.cuda.empty_cache()

                store_ply(ply_file_path, point_cloud.xyz, point_cloud.rgb)
                with open("{}.config.json".format(ply_file_path), "w") as f:
                    f.write(params_json)
                print("Point cloud saved to '{}'".format(ply_file_path))
        else:
            point_cloud = None
        return ImageSet(
            image_names=image_names,
            image_paths=image_paths,
            mask_paths=None,
            cameras=cameras,
            extra_data=depth_paths,
            extra_data_processor=self.get_depth_map_processor(self.params.scale, self.params.depth_scale, self.params.max_depth, self.params.use_inverse_depth) if self.params.use_depth else self.return_none
        ), point_cloud

    def get_outputs(self) -> DataParserOutputs:
        train_set, point_cloud = self._parse_json(self.params.train, True)
        test_set, _ = self._parse_json(self.params.test)

        return DataParserOutputs(
            train_set=train_set,
            val_set=test_set,
            test_set=test_set,
            point_cloud=point_cloud,
            appearance_group_ids=None,
        )

    @staticmethod
    def get_depth_map_processor(scale, depth_scale, max_depth, inverse: bool):
        os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
        import cv2

        depth_value_scale_factor = scale * depth_scale

        def read_depth_map(path):
            depth_map = cv2.imread(
                path,
                cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH,
            )[..., 0] * depth_value_scale_factor
            depth_map_mask = depth_map < max_depth * depth_value_scale_factor

            depth_map = torch.tensor(depth_map, dtype=torch.float)
            depth_map_mask = torch.tensor(depth_map_mask, dtype=torch.bool)

            if inverse:
                depth_map = torch.where(
                    depth_map > 0.,
                    1. / depth_map,
                    depth_map.max(),
                )

            return depth_map, depth_map_mask

        return read_depth_map

    @staticmethod
    def return_none(*args, **kwargs):
        return None
