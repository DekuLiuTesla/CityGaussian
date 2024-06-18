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

        return grayscale_factors.to(torch.float), gamma.to(torch.float)
