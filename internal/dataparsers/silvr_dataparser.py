from dataclasses import dataclass
from typing import Literal
import os
import json
import numpy as np
import torch
from .dataparser import ImageSet, PointCloud, DataParserConfig, DataParser, DataParserOutputs
from .blender_dataparser import BlenderDataParser


@dataclass
class SILVR(DataParserConfig):
    white_background: bool = False

    n_random_points: int = 100_000

    random_point_color: bool = False

    random_point_range: float = 10.

    split_mode: Literal["reconstruction"] = "reconstruction"

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "SILVRDataParser":
        return SILVRDataParser(path, output_path, global_rank, self)


class SILVRDataParser(DataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: SILVR) -> None:
        super().__init__()
        self.path = path
        self.output_path = output_path
        self.global_rank = global_rank
        self.params = params

    def get_outputs(self) -> DataParserOutputs:
        with open(os.path.join(self.path, "transforms.json"), "r") as f:
            transforms = json.load(f)

        train_set = BlenderDataParser.parse_transforms(transforms=transforms, path=self.path)
        transforms["frames"] = transforms["frames"][:1]
        val_set = BlenderDataParser.parse_transforms(transforms=transforms, path=self.path)

        num_pts = self.params.n_random_points
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * self.params.random_point_range - self.params.random_point_range / 2. + torch.mean(train_set.cameras.camera_center, dim=0).numpy()
        if self.params.random_point_color is True:
            rgb = np.asarray(np.random.random((num_pts, 3)) * 255, dtype=np.uint8)  # random rgb color will produce artifacts
        else:
            rgb = np.ones((num_pts, 3), dtype=np.uint8) * 127

        return DataParserOutputs(
            train_set=train_set,
            val_set=val_set,
            test_set=val_set,
            point_cloud=PointCloud(
                xyz=xyz,
                rgb=rgb,
            ),
            appearance_group_ids=None,
        )
