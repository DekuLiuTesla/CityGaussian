from dataclasses import dataclass


@dataclass
class AppearanceModelOptimizationParams:
    lr: float = 1e-3
    eps: float = 1e-15
    gamma: float = 1
    max_steps: int = 30_000


@dataclass
class AppearanceModelParams:
    optimization: AppearanceModelOptimizationParams

    n_grayscale_factors: int = 3
    n_gammas: int = 3
    n_neurons: int = 32
    n_hidden_layers: int = 2
    n_frequencies: int = 4
    grayscale_factors_activation: str = "Sigmoid"
    gamma_activation: str = "Softplus"

@dataclass
class SwagAppearanceModelParams:
    optimization: AppearanceModelOptimizationParams

    n_appearance_count: int=6000
    n_appearance_dims: int = 24
    n_input_dims: int = 30
    n_neurons: int = 64
    n_hidden_layers: int = 3
    color_activation: str = "Sigmoid"

@dataclass
class VastAppearanceModelParams:
    optimization: AppearanceModelOptimizationParams

    n_appearance_count: int=6000
    n_appearance_dims: int = 64
    n_rgb_dims: int = 3
    std: float = 1e-4