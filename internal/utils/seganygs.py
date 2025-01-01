from typing import Union, List
import torch
import hdbscan
import numpy as np


class ScaleGateUtils:
    def __init__(self, scale_gate: torch.nn.Module):
        self.scale_gate = scale_gate
        self._device = next(scale_gate.parameters()).device

    def __call__(self, scale):
        return self.scale_gate(
            torch.tensor([scale], dtype=torch.float, device=self._device)
        )


class SegAnyGSUtils:
    @classmethod
    def get_pca_projection_matrix(cls, semantic_features: torch.Tensor, n_components: int = 3, seed: int = 42):
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        randint = torch.randint(0, semantic_features.shape[0], [200_000], generator=generator)
        X = semantic_features[randint, :]

        n = X.shape[0]
        mean = torch.mean(X, dim=0)
        X = X - mean
        covariance_matrix = (1 / n) * torch.matmul(X.T, X).float()  # An old torch bug: matmul float32->float16,
        eigenvalues, eigenvectors = torch.linalg.eig(covariance_matrix)
        idx = torch.argsort(-eigenvalues.real)
        eigenvectors = eigenvectors.real[:, idx]
        proj_mat = eigenvectors[:, 0:n_components]

        return proj_mat

    @classmethod
    def get_scale_conditioned_semantic_features(cls, semantic_features: torch.Tensor, scale_gate: ScaleGateUtils, scale: float):
        return torch.nn.functional.normalize(
            semantic_features * scale_gate(scale).to(semantic_features.device),
            dim=-1,
        )

    @classmethod
    def get_pca_projected_colors(cls, semantic_features, pca_projection_matrix):
        colors = (semantic_features @ pca_projection_matrix)
        colors = colors - colors.min(dim=0).values
        colors = colors / (colors.max(dim=0).values + 1e-6)
        return colors

    @classmethod
    def cluster_label2colors(cls, seg_score):
        label_to_color = np.random.rand(1000, 3)
        point_colors = label_to_color[seg_score.argmax(dim=-1).cpu().numpy()]
        point_colors[seg_score.max(dim=-1)[0].detach().cpu().numpy() < 0.5] = (0, 0, 0)

        return point_colors

    @classmethod
    def cluster_3d(cls, scale_conditioned_semantic_features: torch.Tensor):
        # select points randomly
        normed_sampled_point_features = scale_conditioned_semantic_features[torch.rand(scale_conditioned_semantic_features.shape[0]) > 0.98]

        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, cluster_selection_epsilon=0.01)
        cluster_labels = clusterer.fit_predict(normed_sampled_point_features.detach().cpu().numpy())
        # print(np.unique(cluster_labels))

        cluster_centers = torch.zeros(len(np.unique(cluster_labels)) - 1, normed_sampled_point_features.shape[-1])
        for i in range(1, len(np.unique(cluster_labels))):
            cluster_centers[i - 1] = torch.nn.functional.normalize(normed_sampled_point_features[cluster_labels == i - 1].mean(dim=0), dim=-1)

        seg_score = torch.einsum('nc,bc->bn', cluster_centers.cpu(), scale_conditioned_semantic_features.cpu())

        point_colors = cls.cluster_label2colors(seg_score=seg_score)

        return cluster_labels, cluster_centers, seg_score, point_colors

    @classmethod
    def cluster_3d_as_dict(cls, scale_conditioned_semantic_features: torch.Tensor):
        cluster_result = cls.cluster_3d(scale_conditioned_semantic_features)
        return {
            "cluster_labels": cluster_result[0],
            "cluster_centers": cluster_result[1],
            "seg_score": cluster_result[2],
            "point_colors": cluster_result[3],
        }

    @classmethod
    def get_similarities(
            cls,
            scale_conditioned_semantic_features: torch.Tensor,
            scale_conditioned_query_features: torch.Tensor,
    ):
        """
        :param scale_conditioned_semantic_features: [N_points, N_feature_dims]
        :param scale_conditioned_query_features:  [N_features, N_feature_dims]
        :return: [N_points, N_features]
        """

        return (torch.einsum("NC,CA->NA", scale_conditioned_semantic_features, scale_conditioned_query_features.T) + 1.) / 2

    @classmethod
    def get_similarities_by_raw_feature_list(
            cls,
            scale_conditioned_semantic_features: torch.Tensor,
            query_features: List,
            scale_gate: ScaleGateUtils,
            scale: float,
    ):
        scale_conditioned_query_features = cls.get_scale_conditioned_query_features(
            query_features,
            scale_gate,
            scale,
        )
        return cls.get_similarities(scale_conditioned_semantic_features, scale_conditioned_query_features)

    @classmethod
    def get_scale_conditioned_query_features(cls, query_features: Union[torch.Tensor, List], scale_gate: ScaleGateUtils, scale):
        """
        :param query_features: [N_features, N_feature_dims]
        :param scale_gate:
        :param scale:
        :return:
        """
        if isinstance(query_features, list):
            query_features = torch.stack(query_features)
        scale_gated_query_feature = query_features * scale_gate(scale).to(device=query_features.device)
        scale_gated_query_feature = torch.nn.functional.normalize(scale_gated_query_feature, dim=-1)
        return scale_gated_query_feature

    @classmethod
    def get_segment_mask_by_raw_feature_list(
            cls,
            scale_conditioned_semantic_features: torch.Tensor,
            query_features: List,
            scale_gate: ScaleGateUtils,
            scale: float,
            score: float,
            gamma: float = None,
            gamma_eps: float = 1e-9,
            return_similarity_matrix: bool = False,
    ):
        similarities = cls.get_similarities_by_raw_feature_list(
            scale_conditioned_semantic_features,
            query_features,
            scale_gate,
            scale,
        )
        if gamma is not None and gamma != 1.:
            similarities = torch.pow(similarities + gamma_eps, gamma)

        mask = (similarities >= score).sum(dim=-1) > 0

        if return_similarity_matrix is True:
            return mask, similarities
        return mask
