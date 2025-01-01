from dataclasses import dataclass, field
from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel, OptimizationConfig


@dataclass
class VanillaGaussianWithSparseAdam(VanillaGaussian):
    optimization: OptimizationConfig = field(default_factory=lambda: OptimizationConfig(
        optimizer={"class_path": "SparseGaussianAdam"}
    ))

    def instantiate(self, *args, **kwargs) -> "VanillaGaussianModel":
        return VanillaGaussianModel(self)
