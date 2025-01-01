from dataclasses import dataclass
from typing import Dict

import torch

from .vanilla_gaussian import VanillaGaussian, VanillaGaussianModel


@dataclass
class Gaussian2D(VanillaGaussian):
    def instantiate(self, *args, **kwargs) -> "Gaussian2DModel":
        return Gaussian2DModel(self)


class Gaussian2DModelMixin:
    def before_setup_set_properties_from_pcd(self, xyz: torch.Tensor, rgb: torch.Tensor, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_pcd(
            xyz=xyz,
            rgb=rgb,
            property_dict=property_dict,
            *args,
            **kwargs,
        )
        with torch.no_grad():
            property_dict["scales"] = property_dict["scales"][..., :2]
            # key to a quality comparable to hbb1/2d-gaussian-splatting
            property_dict["rotations"].copy_(torch.rand_like(property_dict["rotations"]))

    def before_setup_set_properties_from_number(self, n: int, property_dict: Dict[str, torch.Tensor], *args, **kwargs):
        super().before_setup_set_properties_from_number(
            n=n,
            property_dict=property_dict,
            *args,
            **kwargs,
        )
        property_dict["scales"] = property_dict["scales"][..., :2]


class Gaussian2DModel(Gaussian2DModelMixin, VanillaGaussianModel):
    pass
