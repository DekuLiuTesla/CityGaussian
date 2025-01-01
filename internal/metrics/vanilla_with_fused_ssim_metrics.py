from dataclasses import dataclass
from .vanilla_metrics import VanillaMetrics, VanillaMetricsImpl


@dataclass
class VanillaWithFusedSSIMMetrics(VanillaMetrics):
    fused_ssim: bool = True

    def instantiate(self, *args, **kwargs) -> "VanillaWithFusedSSIMMetricsModule":
        return VanillaWithFusedSSIMMetricsModule(self)


class VanillaWithFusedSSIMMetricsModule(VanillaMetricsImpl):
    pass
