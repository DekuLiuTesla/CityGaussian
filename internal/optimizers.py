from typing import Tuple
from dataclasses import dataclass
from typing import Any
import torch
from internal.configs.instantiate_config import InstantiatableConfig


@dataclass
class OptimizerConfig(InstantiatableConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        raise NotImplementedError()


@dataclass
class Adam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        return torch.optim.Adam(
            params,
            lr,
            *args,
            **kwargs,
        )


@dataclass
class SelectiveAdam(OptimizerConfig):
    betas: Tuple[float, float] = (0.9, 0.999)

    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        for group in params:
            if "lr" not in group:
                group["lr"] = lr

        from gsplat.optimizers import SelectiveAdam
        from torch.optim.optimizer import _use_grad_for_differentiable

        class Adapter(SelectiveAdam):
            def on_after_backward(self, outputs, batch, gaussian_model, global_step, pl_module):
                self.visibility = outputs["viewspace_points"].has_hit_any_pixels

            @_use_grad_for_differentiable
            def step(self, closure=None):
                self._cuda_graph_capture_health_check()

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                super().step(self.visibility)

                return loss

        return Adapter(
            params,
            betas=self.betas,
            *args,
            **kwargs,
        )


@dataclass
class SparseGaussianAdam(OptimizerConfig):
    def instantiate(self, params, lr: float, *args, **kwargs) -> Any:
        from diff_accel_gaussian_rasterization import SparseGaussianAdam
        from torch.optim.optimizer import _use_grad_for_differentiable

        class Adapter(SparseGaussianAdam):
            def on_after_backward(self, outputs, batch, gaussian_model, global_step, pl_module):
                self.visibility = outputs["visibility_filter"]

            @_use_grad_for_differentiable
            def step(self, closure=None):
                self._cuda_graph_capture_health_check()

                loss = None
                if closure is not None:
                    with torch.enable_grad():
                        loss = closure()

                super().step(self.visibility, self.visibility.shape[0])

                return loss

        return Adapter(
            params,
            lr,
            *args,
            **kwargs,
        )
