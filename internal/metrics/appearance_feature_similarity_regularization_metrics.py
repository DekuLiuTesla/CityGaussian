"""
This regularization can encourage greater similarity among the features of neighboring Gaussians.
"""

from dataclasses import dataclass, field
import time
import torch
from pytorch3d.ops import knn_points

from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl

class Similarity:
    @staticmethod
    def get_similarities(features: torch.Tensor):
        """
        Return: The larger the value, the higher the similarity
        """
        raise NotImplementedError()

@dataclass
class Cosine(Similarity):
    @staticmethod
    def get_similarities(features: torch.Tensor):
        normalized_appearance_features = torch.nn.functional.normalize(
            features,
            dim=-1,
        )  # [N_samples, N_k, N_feature_dims]

        similarities = (normalized_appearance_features[:, :, None, :] * normalized_appearance_features[:, None, :, :]).sum(dim=-1)  # [N_samples, N_k, N_k]

        return similarities

@dataclass
class Euclidean(Similarity):
    @staticmethod
    def get_similarities(features: torch.Tensor):
        distances = torch.norm(features[:, :, None, :] - features[:, None, :, :], dim=-1)
        return -distances


@dataclass
class SimilarityRegularizationConfigMixin:
    n_appearance_samples: int = 51_200

    n_appearance_nn: int = 16

    distance_weight_decay: float = 200.
    """
    Used to decrease the weights with increasing distance.
    The value should be higher for smaller scenes.
    """

    similarity_reg_from: int = 0

    similarity_reg_lambda: float = 0.2

    similarity_reg_interval: int = 50

    similarity_type: Similarity = field(default_factory=lambda: {"class_path": "Cosine"})


class SimilarityRegularizationMixin:
    def get_similarity_metric(self, gaussian_model):
        started_at = time.time()
        with torch.no_grad():
            means = gaussian_model.get_means()

            sample_indices = torch.randperm(means.shape[0], device=means.device)[:self.config.n_appearance_samples]

            knn_results = knn_points(
                means[sample_indices].unsqueeze(0),
                means.unsqueeze(0),
                K=self.config.n_appearance_nn,
            )
            knn_consumed_time = time.time() - started_at

            distance_based_weight = torch.exp(-self.config.distance_weight_decay * knn_results.dists.squeeze())  # [N_samples, N_k]

        appearance_features = gaussian_model.get_appearance_features()[knn_results.idx.squeeze()]

        raw_similarities_reg = -self.config.similarity_type.get_similarities(appearance_features)  # flip the direction of the optimization to the higher values

        weighted_similarities_reg = raw_similarities_reg * distance_based_weight[:, None, :]

        # upper
        triu_mask = torch.triu(torch.ones(raw_similarities_reg.shape[1:3], dtype=torch.bool, device=raw_similarities_reg.device), diagonal=1)

        valid_similarities = weighted_similarities_reg[:, triu_mask]

        similarities_reg = valid_similarities.mean()

        return similarities_reg * self.config.similarity_reg_lambda, (
            sample_indices,
            knn_results,
            raw_similarities_reg,
            distance_based_weight,
            raw_similarities_reg,
            weighted_similarities_reg,
            triu_mask,
            valid_similarities,
            similarities_reg,
        ), knn_consumed_time

    def similarity_metric_backward_hook(self, outputs, batch, gaussian_model, step, pl_module):
        if step < self.config.similarity_reg_from:
            return
        if step % self.config.similarity_reg_interval != 0:
            return

        similarities_reg, meta, _ = self.get_similarity_metric(gaussian_model)
        pl_module.manual_backward(similarities_reg)
        pl_module.log("train/as_r", meta[-1], prog_bar=True, on_step=False, on_epoch=True, batch_size=pl_module.batch_size)

    def training_setup(self, pl_module):
        pl_module.on_after_backward_hooks.append(self.similarity_metric_backward_hook)
        return super().training_setup(pl_module)


@dataclass
class VanillaMetricsWithSimilarityRegularization(SimilarityRegularizationConfigMixin, VanillaMetrics):
    def instantiate(self, *args, **kwargs) -> "VanillaMetricsWithSimilarityRegularizationModule":
        return VanillaMetricsWithSimilarityRegularizationModule(self)


class VanillaMetricsWithSimilarityRegularizationModule(SimilarityRegularizationMixin, VanillaMetricsImpl):
    pass
