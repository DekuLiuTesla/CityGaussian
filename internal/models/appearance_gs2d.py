from dataclasses import dataclass
from .appearance_feature_gaussian import AppearanceFeatureGaussian, AppearanceFeatureGaussianModel
from .gaussian_2d import Gaussian2D, Gaussian2DModelMixin


@dataclass
class AppearanceGS2D(AppearanceFeatureGaussian):
    def instantiate(self, *args, **kwargs) -> "AppearanceGS2dModel":
        return AppearanceGS2dModel(self)


class AppearanceGS2dModel(Gaussian2DModelMixin, AppearanceFeatureGaussianModel):
    pass
