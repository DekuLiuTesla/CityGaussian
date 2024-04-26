"""
F_θ of the paper "SWAG: Splatting in the Wild images with Appearance-conditioned Gaussians" (https://arxiv.org/abs/2403.10427):
    (c^I, Δα^I) = F_θ(c, emb(x), l_I)
"""

from dataclasses import dataclass
from typing import Literal, Union
import torch
from torch import nn
from internal.utils.network_factory import NetworkFactory
from internal.configs.tcnn_encoding_config import TCNNEncodingConfig


@dataclass
class GridEncodingConfig(TCNNEncodingConfig):
    type: Literal["hashgrid", "densegrid"] = "hashgrid"

    n_levels = 12


@dataclass
class NetworkConfig:
    n_neurons: int = 64
    n_layers: int = 3


@dataclass
class EmbeddingConfig:
    num_embeddings: int = 2048
    embedding_dim: int = 24


@dataclass
class GridEncodingOptimizationConfig:
    lr: float = 2.5e-3
    max_steps: int = 60_000
    lr_final_factor: float = 0.01
    eps: float = 1e-15


@dataclass
class EmbeddingOptimizationConfig:
    lr: float = 2.5e-3
    max_steps: int = 60_000
    lr_final_factor: float = 0.01
    eps: float = 1e-15


@dataclass
class OptimizationConfig:
    gird_encoding: GridEncodingOptimizationConfig = GridEncodingOptimizationConfig()

    embedding: EmbeddingOptimizationConfig = EmbeddingOptimizationConfig()


class SWAGModel(nn.Module):
    def __init__(
            self,
            network: NetworkConfig = NetworkConfig(),
            grid_encoding: GridEncodingConfig = GridEncodingConfig(),
            embedding: EmbeddingConfig = EmbeddingConfig(),
    ) -> None:
        super().__init__()

        self.network_config = network
        self.grid_encoding_config = grid_encoding
        self.embedding_config = embedding

        self._setup()

    def _setup(self):
        import tinycudann as tcnn

        grid_encoding_input_dims = 3
        self.grid_encoding = tcnn.Encoding(
            n_input_dims=grid_encoding_input_dims,
            encoding_config=self.grid_encoding_config.get_encoder_config(grid_encoding_input_dims),
        )

        self.image_embedding = nn.Embedding(self.embedding_config.num_embeddings, self.embedding_config.embedding_dim)

        network_factory = NetworkFactory(tcnn=False)
        self.theta = network_factory.get_network(
            n_input_dims=self.grid_encoding.n_output_dims + self.embedding_config.embedding_dim + 3,
            n_output_dims=4,  # 3-c^I, 1-Δα^I
            n_layers=self.network_config.n_layers,
            n_neurons=self.network_config.n_neurons,
            activation="ReLU",
            output_activation="None",
        )

    def forward(
            self,
            colors: torch.Tensor,  # [n, 3]
            x: torch.Tensor,  # [n, 3], normalized 3D means of the Gaussian
            image_id: Union[int, torch.Tensor],
    ):
        x_emb = self.grid_encoding(x).to(colors.dtype)
        image_emb = self.image_embedding(torch.tensor([image_id], dtype=torch.int, device=colors.device)).repeat((colors.shape[0], 1))

        input = torch.concat([colors, x_emb, image_emb], dim=-1)
        output = self.theta(input)

        # (c^I, Δα^I)
        return nn.functional.sigmoid(output[:, :3]), output[:, -1]
