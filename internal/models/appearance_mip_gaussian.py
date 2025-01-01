from dataclasses import dataclass

from .appearance_feature_gaussian import AppearanceFeatureGaussian, AppearanceFeatureGaussianModel
from .mip_splatting import MipSplattingConfigMixin, MipSplattingModelMixin


@dataclass
class AppearanceMipGaussian(MipSplattingConfigMixin, AppearanceFeatureGaussian):
    def instantiate(self, *args, **kwargs) -> "AppearanceMipGaussianModel":
        return AppearanceMipGaussianModel(self)


class AppearanceMipGaussianModel(MipSplattingModelMixin, AppearanceFeatureGaussianModel):
    config: AppearanceMipGaussian
