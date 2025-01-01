import os
import json
import numpy as np
import torch
from dataclasses import dataclass
from .colmap_dataparser import Colmap, ColmapDataParser
from internal.dataparsers import DataParserOutputs


@dataclass
class EstimatedDepthColmap(Colmap):
    depth_dir: str = "estimated_depths"

    depth_rescaling: bool = True

    depth_scale_name: str = "estimated_depth_scales"

    depth_scale_lower_bound: float = 0.2

    depth_scale_upper_bound: float = 5.

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "EstimatedDepthColmapDataParser":
        return EstimatedDepthColmapDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class EstimatedDepthColmapDataParser(ColmapDataParser):
    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        if self.params.depth_rescaling is True:
            with open(os.path.join(self.path, self.params.depth_scale_name + ".json"), "r") as f:
                depth_scales = json.load(f)

            median_scale = np.median(np.asarray([i["scale"] for i in depth_scales.values()]))

        loaded_depth_count = 0
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                depth_file_path = os.path.join(self.path, self.params.depth_dir, f"{image_name}.npy")
                if os.path.exists(depth_file_path) is False:
                    print("[WARNING] {} does not have a depth file".format(image_name))
                    continue

                depth_scale = {
                    "scale": 1.,
                    "offset": 0.,
                }
                if self.params.depth_rescaling is True:
                    depth_scale = depth_scales.get(image_name, None)
                    if depth_scale is None:
                        print("[WARNING {} does not have a depth scale]".format(image_name))
                        continue
                    if depth_scale["scale"] < self.params.depth_scale_lower_bound * median_scale or depth_scale["scale"] > self.params.depth_scale_upper_bound * median_scale:
                        print("[WARNING depth scale of {} out of bound]".format(image_name))
                        continue
                
                image_set.extra_data[idx] = (depth_file_path, depth_scale)
                loaded_depth_count += 1
            image_set.extra_data_processor = self.load_depth

        assert loaded_depth_count > 0
        print("found {} depth maps".format(loaded_depth_count))

        return dataparser_outputs

    @staticmethod
    def load_depth(depth_info):
        if depth_info is None:
            return None

        depth_file_path, depth_scale = depth_info
        depth = np.load(depth_file_path) * depth_scale["scale"] + depth_scale["offset"]

        return torch.tensor(depth, dtype=torch.float)
