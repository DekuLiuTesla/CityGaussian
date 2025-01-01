import os
import torch
import csv
from typing import Tuple
from dataclasses import dataclass
from internal.dataparsers import DataParser, DataParserOutputs
from internal.dataparsers.colmap_dataparser import Colmap, ColmapDataParser


@dataclass
class PhotoTourism(Colmap):
    semantic_feature: bool = False

    semantic_feature_dir: str = "SD"

    def instantiate(self, path: str, output_path: str, global_rank: int) -> DataParser:
        return PhotoTourismDataParser(path, output_path, global_rank, self)


class PhotoTourismDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: PhotoTourism) -> None:
        super().__init__(path, output_path, global_rank, params)

        self.tsv_file_path = None
        for i in os.scandir(self.path):
            if i.name.endswith(".tsv") is True:
                self.tsv_file_path = i.path

        assert self.tsv_file_path is not None, "tsv file not found in {}, please download one from the 'additional links' section of https://nerf-w.github.io/, or create one yourself.".format(self.path)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        if self.params.semantic_feature:
            from .spotless_colmap_dataparser import SpotLessColmapDataParser

            for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
                for idx, image_name in enumerate(image_set.image_names):
                    image_name_without_ext = image_name[:image_name.rfind(".")]
                    semantic_file_name = f"{image_name_without_ext}.npy"
                    image_set.extra_data[idx] = os.path.join(self.path, "dense", self.params.semantic_feature_dir, semantic_file_name)
                image_set.extra_data_processor = SpotLessColmapDataParser.read_semantic_feature

        return dataparser_outputs

    def detect_sparse_model_dir(self) -> str:
        return os.path.join(self.path, "dense", "sparse")

    def get_image_dir(self) -> str:
        if self.params.image_dir is None:
            image_dir = os.path.join(self.path, "dense", "images")
            if self.params.down_sample_factor > 1:
                image_dir = image_dir + "_{}".format(self.params.down_sample_factor)
            return image_dir
        return os.path.join(self.path, self.params.image_dir)

    def build_split_indices(self, image_name_list) -> Tuple[list, list]:
        print("load {}".format(self.tsv_file_path))

        training_set_filenames = {}
        validation_set_filenames = {}
        with open(self.tsv_file_path) as fd:
            rd = csv.reader(fd, delimiter="\t", quotechar='"')
            for row in rd:
                if row[2] == "train":
                    training_set_filenames[row[0]] = True
                else:
                    validation_set_filenames[row[0]] = True

        training_set_indices = []
        validation_set_indices = []
        for idx, image_name in enumerate(image_name_list):
            if image_name in training_set_filenames:
                training_set_indices.append(idx)
            elif image_name in validation_set_filenames:
                validation_set_indices.append(idx)
                if self.params.split_mode == "reconstruction":
                    training_set_indices.append(idx)

        return training_set_indices, validation_set_indices
