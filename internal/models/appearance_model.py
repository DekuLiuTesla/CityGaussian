from typing import Union
import torch
import torch.nn.functional as F

from torch import nn


class AppearanceModel(nn.Module):
    def __init__(
            self,
            n_input_dims: int = 1,
            n_grayscale_factors: int = 3,
            n_gammas: int = 1,
            n_neurons: int = 32,
            n_hidden_layers: int = 2,
            n_frequencies: int = 4,
            grayscale_factors_activation: str = "Sigmoid",
            gamma_activation: str = "Softplus",
    ) -> None:
        super().__init__()

        self.device_indicator = nn.Parameter(torch.empty(0))

        # create model
        import tinycudann as tcnn
        self.grayscale_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_grayscale_factors,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": grayscale_factors_activation,
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
        )
        self.gamma_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_gammas,
            encoding_config={
                "otype": "Frequency",
                "n_frequencies": n_frequencies,
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": gamma_activation,
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
        )

    def forward(self, x):
        grayscale_factors = self.grayscale_model(x).reshape((x.shape[0], -1, 1, 1))
        gamma = self.gamma_model(x).reshape((x.shape[0], -1, 1, 1))

        return grayscale_factors, gamma

    def get_appearance(self, x: Union[float, torch.Tensor]):
        model_input = torch.tensor([[x]], dtype=torch.float16, device=self.device_indicator.device)
        grayscale_factors, gamma = self(model_input)
        grayscale_factors = grayscale_factors.reshape((-1, 1, 1))
        gamma = gamma.reshape((-1, 1, 1))

        return grayscale_factors, gamma


class SwagAppearanceModel(nn.Module):
    def __init__(
            self,
            n_appearance_count: int = 6000,
            n_appearance_dims: int = 24,
            n_input_dims: int = 30,
            n_output_dims: int = 3,
            n_neurons: int = 64,
            n_hidden_layers: int = 3,
            color_activation: str = "Sigmoid",
    ) -> None:
        super().__init__()

        self.device_indicator = nn.Parameter(torch.empty(0))
        self.embedding_a = nn.Embedding(n_appearance_count, n_appearance_dims)

        # create model
        import tinycudann as tcnn
        self.appearance_model = tcnn.NetworkWithInputEncoding(
            n_input_dims=n_input_dims,
            n_output_dims=n_output_dims,
            encoding_config={
                "otype": "Composite",
                "nested": [
                    {
                        "n_dims_to_encode": 3, # XYZ
                        "otype": "Grid",
                        "type": "Hash",
                        "n_levels": 12, 
                        "n_features_per_level": 2,
                        "log2_hashmap_size": 19,
                        "base_resolution": 16,
                        "per_level_scale": 1.5544,  # Ensure finest resolution is 2048
                    },
                    {
                        # Number of remaining linear dims is automatically derived
                        "otype": "Identity"
                    }
                ]
            },
            network_config={
                "otype": "FullyFusedMLP",
                "activation": "ReLU",
                "output_activation": color_activation,
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
        )

    def forward(self, xyz, color, appearance_id):
        embedding_a = self.embedding_a(appearance_id)
        input = torch.cat([xyz, color, embedding_a], dim=-1)
        out_color = self.appearance_model(input)

        return out_color

    def get_appearance(self, xyz, color, appearance_id):
        embedding_a = self.embedding_a(appearance_id)[None, :].repeat(xyz.shape[0], 1)
        input = torch.cat([xyz, color, embedding_a], dim=-1)
        out_color = self.appearance_model(input)

        return out_color