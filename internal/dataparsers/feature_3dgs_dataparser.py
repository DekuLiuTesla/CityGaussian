import os
import torch

from dataclasses import dataclass
from . import DataParserOutputs, DataParser
from .colmap_dataparser import Colmap, ColmapDataParser


@dataclass
class Feature3DGSColmap(Colmap):
    feature_dir: str = "semantic/sam_features"

    filename_suffix: str = ""

    filename_include_image_ext: bool = True

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return Feature3DGSColmapDataParser(path=path, output_path=output_path, global_rank=global_rank, params=self)


class Feature3DGSColmapDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: Feature3DGSColmap) -> None:
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        # val_set and test_set are same object
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                if self.params.filename_include_image_ext is False:
                    image_name = image_name[:image_name.rfind(".")]
                semantic_file_name = f"{image_name}{self.params.filename_suffix}.pt"
                image_set.extra_data[idx] = os.path.join(self.path, self.params.feature_dir, semantic_file_name)
            image_set.extra_data_processor = Feature3DGSColmapDataParser.read_semantic_data

        # remove image paths to avoid caching
        for i in [dataparser_outputs.train_set, dataparser_outputs.val_set, dataparser_outputs.test_set]:
            for j in range(len(i.image_paths)):
                i.image_paths[j] = None

        return dataparser_outputs

    @staticmethod
    def read_semantic_data(path):
        return torch.load(path, map_location="cpu")
