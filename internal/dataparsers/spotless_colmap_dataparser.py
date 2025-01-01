import os
import numpy as np
import torch

from dataclasses import dataclass
from typing import Literal, Tuple
from . import DataParserOutputs
from .colmap_dataparser import Colmap, ColmapDataParser


@dataclass
class SpotLessColmap(Colmap):
    semantic_feature_dir: str = "SD"

    train_keyword: str = "clutter"

    test_keyword: str = "extra"

    split_mode: Literal["reconstruction", "experiment", "keyword"] = "keyword"

    cluster: bool = False

    def instantiate(self, path: str, output_path: str, global_rank: int) -> "SpotLessColmapDataParser":
        return SpotLessColmapDataParser(path, output_path, global_rank, self)


class SpotLessColmapDataParser(ColmapDataParser):
    def __init__(self, path: str, output_path: str, global_rank: int, params: SpotLessColmap) -> None:
        super().__init__(path, output_path, global_rank, params)

    def get_outputs(self) -> DataParserOutputs:
        dataparser_outputs = super().get_outputs()

        extra_data_processor = SpotLessColmapDataParser.read_semantic_feature
        if self.params.cluster is True:
            extra_data_processor = SpotLessColmapDataParser.read_semantic_feature_and_cluster

        # val_set and test_set are same object
        for image_set in [dataparser_outputs.train_set, dataparser_outputs.val_set]:
            for idx, image_name in enumerate(image_set.image_names):
                image_name_without_ext = image_name[:image_name.rfind(".")]
                semantic_file_name = f"{image_name_without_ext}.npy"
                image_set.extra_data[idx] = os.path.join(self.path, self.params.semantic_feature_dir, semantic_file_name)
            image_set.extra_data_processor = extra_data_processor

        return dataparser_outputs

    def build_split_indices(self, image_name_list) -> Tuple[list, list]:
        if self.params.split_mode != "keyword":
            return super().build_split_indices(image_name_list)

        train_indices = []
        test_indices = []

        for idx, image_name in enumerate(image_name_list):
            if image_name.find(self.params.train_keyword) != -1:
                train_indices.append(idx)
            elif image_name.find(self.params.test_keyword) != -1:
                test_indices.append(idx)

        return train_indices, test_indices

    @staticmethod
    def read_semantic_feature(path):
        semantic_feature = np.load(path)
        return torch.tensor(semantic_feature, dtype=torch.float)

    @staticmethod
    def read_semantic_feature_and_cluster(path):
        from sklearn.cluster import AgglomerativeClustering
        from sklearn.neighbors import kneighbors_graph

        feature = np.load(path)

        # cluster
        ft_flat = np.transpose(feature.reshape((1280, 50 * 50)), (1, 0))
        x = np.linspace(0, 1, 50)
        y = np.linspace(0, 1, 50)
        xv, yv = np.meshgrid(x, y)
        indxy = np.reshape(np.stack([xv, yv], axis=-1), (50 * 50, 2))
        knn_graph = kneighbors_graph(indxy, 8, include_self=False)
        model = AgglomerativeClustering(
            linkage="ward", connectivity=knn_graph, n_clusters=100
        )
        model.fit(ft_flat)
        feature = np.array(
            [model.labels_ == i for i in range(model.n_clusters)],
            dtype=np.float32,
        ).reshape((model.n_clusters, 50, 50))

        return torch.tensor(feature, dtype=torch.float)
