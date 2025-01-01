from typing import Union, Any, List, Dict, Tuple, Optional
from abc import ABC, abstractmethod
import torch
from torch import nn

from internal.configs.instantiate_config import InstantiatableConfig
from internal.utils.gaussian_containers import FreezableParameterDict


class GaussianModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.gaussians = self.setup_gaussians_container()

    @staticmethod
    def setup_gaussians_container():
        return nn.ParameterDict()

    @abstractmethod
    def get_property_names(self) -> Tuple[str, ...]:
        raise NotImplementedError()

    @property
    def property_names(self) -> Tuple[str, ...]:
        return self.get_property_names()

    def get_property(self, name: str) -> torch.Tensor:
        """Get single raw property"""
        return self.gaussians[name]

    def get_properties(self) -> Dict[str, torch.Tensor]:
        """Get all raw properties as a dict"""
        return {name: self.gaussians[name] for name in self.property_names}

    def set_property(self, name: str, value: torch.Tensor):
        """Set single raw property"""
        self.gaussians[name] = value

    def set_properties(self, properties: Dict[str, torch.Tensor]):
        """
        Set all raw properties.
        This setter will not update optimizers.
        """

        for name in self.property_names:
            self.gaussians[name] = properties[name]

    def update_properties(self, properties: Dict[str, torch.Tensor], strict: bool = True):
        """
        Replace part of the properties by those provided in `properties`
        """
        for name in properties:
            if name not in self.gaussians and strict is True:
                raise RuntimeError("`{}` is not a property".format(name))
            self.gaussians[name] = properties[name]

    @property
    def properties(self) -> Dict[str, torch.Tensor]:
        return self.get_properties()

    @properties.setter
    def properties(self, properties: Dict[str, torch.Tensor]):
        """
        this setter will not update optimizers
        """

        self.set_properties(properties)

    def get_n_gaussians(self) -> int:
        return self.gaussians[next(iter(self.gaussians))].shape[0]

    @property
    def n_gaussians(self) -> int:
        return self.get_n_gaussians()

    def freeze(self):
        self.gaussians = FreezableParameterDict(self.gaussians, new_requires_grad=False)

    @abstractmethod
    def setup_from_pcd(self, xyz, rgb, *args, **kwargs):
        """
        Args:
            xyz: [N, 3]
            rgb: [N, 3], must be normalized
        """

        pass

    @abstractmethod
    def setup_from_number(self, n: int, *args, **kwargs):
        pass

    @abstractmethod
    def setup_from_tensors(self, tensors: Dict[str, torch.Tensor], *args, **kwargs):
        pass

    @abstractmethod
    def training_setup(self, module: "lightning.LightningModule") -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        pass

    def on_train_batch_end(self, step: int, module: "lightning.LightningModule"):
        pass


class Gaussian(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> GaussianModel:
        raise NotImplementedError()


class HasMeanGetter:
    _mean_name = "means"

    def get_means(self) -> torch.Tensor:
        return self.gaussians[self._mean_name]

    @property
    def means(self) -> torch.Tensor:
        return self.gaussians[self._mean_name]

    @means.setter
    def means(self, v):
        self.gaussians[self._mean_name] = v


class HasScaleGetter(ABC):
    _scale_name = "scales"

    @abstractmethod
    def scale_activation(self, scales: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def scale_inverse_activation(self, scales: torch.Tensor) -> torch.Tensor:
        pass

    def get_scales(self) -> torch.Tensor:
        """Return activated scales"""
        return self.scale_activation(self.scales)

    @property
    def scales(self) -> torch.Tensor:
        """Return raw scales"""
        return self.gaussians[self._scale_name]

    @scales.setter
    def scales(self, v):
        """Set raw scales"""
        self.gaussians[self._scale_name] = v


class HasRotationGetter(ABC):
    _rotation_name = "rotations"

    @abstractmethod
    def rotation_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def rotation_inverse_activation(self, rotations: torch.Tensor) -> torch.Tensor:
        pass

    def get_rotations(self) -> torch.Tensor:
        """Return activated rotations"""
        return self.rotation_activation(self.rotations)

    @property
    def rotations(self) -> torch.Tensor:
        """Return raw rotations"""
        return self.gaussians[self._rotation_name]

    @rotations.setter
    def rotations(self, v):
        """Set raw rotations"""
        self.gaussians[self._rotation_name] = v


class HasCovarianceGetter(ABC):
    @abstractmethod
    def get_covariances(self, scaling_modifier: float = 1.):
        pass


class HasOpacityGetter(ABC):
    _opacity_name = "opacities"

    @abstractmethod
    def opacity_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def opacity_inverse_activation(self, opacities: torch.Tensor) -> torch.Tensor:
        pass

    def get_opacities(self) -> torch.Tensor:
        """Return activated opacities"""
        return self.opacity_activation(self.opacities)

    @property
    def opacities(self) -> torch.Tensor:
        """Return raw opacities"""
        return self.gaussians[self._opacity_name]

    @opacities.setter
    def opacities(self, v):
        """Set raw opacities"""
        self.gaussians[self._opacity_name] = v


class HasSHs(ABC):
    _shs_dc_name = "shs_dc"
    _shs_rest_name = "shs_rest"

    # shs_dc

    def get_shs_dc(self) -> torch.Tensor:
        return self.gaussians[self._shs_dc_name]

    @property
    def shs_dc(self) -> torch.Tensor:
        return self.gaussians[self._shs_dc_name]

    @shs_dc.setter
    def shs_dc(self, v):
        self.gaussians[self._shs_dc_name] = v

    # shs_rest

    def get_shs_rest(self) -> torch.Tensor:
        return self.gaussians[self._shs_rest_name]

    @property
    def shs_rest(self) -> torch.Tensor:
        return self.gaussians[self._shs_rest_name]

    @shs_rest.setter
    def shs_rest(self, v):
        self.gaussians[self._shs_rest_name] = v

    # shs

    def get_shs(self) -> torch.Tensor:
        """
        Return: [n, N_SHs, 3]
        """
        return torch.cat((self.shs_dc, self.shs_rest), dim=1)

    # max_sh_degree

    @abstractmethod
    def get_max_sh_degree(self) -> int:
        raise NotImplementedError()

    @property
    def max_sh_degree(self) -> int:
        return self.get_max_sh_degree()

    # active_sh_degree

    @abstractmethod
    def get_active_sh_degree(self) -> int:
        raise NotImplementedError()

    @property
    def active_sh_degree(self) -> int:
        return self.get_active_sh_degree()

    @abstractmethod
    def set_active_sh_degree(self, v):
        raise NotImplementedError()

    @active_sh_degree.setter
    def active_sh_degree(self, v):
        self.set_active_sh_degree(v)


class HasNewGetters(
    HasMeanGetter,
    HasScaleGetter,
    HasRotationGetter,
    HasOpacityGetter,
    HasSHs,
    ABC,
):
    pass


class HasVanillaGetters(ABC):
    @property
    @abstractmethod
    def get_scaling(self):
        pass

    @property
    @abstractmethod
    def get_rotation(self):
        pass

    @property
    @abstractmethod
    def get_xyz(self):
        pass

    @property
    @abstractmethod
    def get_features(self):
        pass

    @property
    @abstractmethod
    def get_opacity(self):
        pass

    @abstractmethod
    def get_covariance(self, scaling_modifier: float = 1.):
        pass
