from dataclasses import dataclass
from lightning import LightningModule
from .density_controller import DensityController, DensityControllerImpl


@dataclass
class StaticDensityController(DensityController):
    def instantiate(self, *args, **kwargs) -> DensityControllerImpl:
        return StaticDensityControllerImpl(self)


class StaticDensityControllerImpl(DensityControllerImpl):
    pass
