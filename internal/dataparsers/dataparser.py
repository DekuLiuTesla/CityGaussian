from typing import Tuple, Optional, Callable
from dataclasses import dataclass

import numpy as np
import torch

from internal.cameras.cameras import Cameras
from internal.configs.instantiate_config import InstantiatableConfig


@dataclass
class ImageSet:
    image_names: list

    image_paths: list
    """ Full path to the image file """

    cameras: Cameras
    """ Camera intrinscis and extrinsics """

    depth_paths: Optional[list] = None
    """ Full path to the depth file """

    mask_paths: Optional[list] = None
    """ Full path to the mask file """

    extra_data: Optional[list] = None

    extra_data_processor: Optional[Callable] = None

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        return self.image_names[index], self.image_paths[index], self.mask_paths[index], self.cameras[index], self.extra_data[index]

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __post_init__(self):
        if self.mask_paths is None:
            self.mask_paths = [None for _ in range(len(self.image_paths))]
        if self.extra_data is None:
            self.extra_data = [None for _ in range(len(self.image_paths))]
        if self.extra_data_processor is None:
            self.extra_data_processor = ImageSet._return_input

    @staticmethod
    def _return_input(i):
        return i

        if self.extra_data_processor is None:
            self.extra_data_processor = ImageSet._return_input

    @staticmethod
    def _return_input(i):
        return i

@dataclass
class PointCloud:
    xyz: np.ndarray  # float

    rgb: np.ndarray  # uint8, in [0, 255]


@dataclass
class DataParserOutputs:
    train_set: ImageSet

    val_set: ImageSet

    test_set: ImageSet

    point_cloud: PointCloud

    # ply_path: str

    appearance_group_ids: Optional[dict] = None

    camera_extent: Optional[float] = None

    sampler: Optional[torch.utils.data.Sampler] = None

    def __post_init__(self):
        if self.camera_extent is None:
            camera_centers = self.train_set.cameras.camera_center
            average_camera_center = torch.mean(camera_centers, dim=0)
            camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
            max_distance = torch.max(camera_distance)
            self.camera_extent = float(max_distance * 1.1)


class DataParser:
    def get_outputs(self) -> DataParserOutputs:
        """
        :return: [training set, validation set, point cloud]
        """

        pass


@dataclass
class DataParserConfig(InstantiatableConfig):
    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        pass
