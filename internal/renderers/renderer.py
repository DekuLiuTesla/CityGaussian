import lightning
import torch
from typing import Any, Union, List, Tuple, Optional, Dict, Callable
from internal.cameras.cameras import Camera
from internal.models.gaussian_model import GaussianModel


class Renderer(torch.nn.Module):
    def forward(
            self,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        pass

    def training_forward(
            self,
            step: int,
            module: lightning.LightningModule,
            viewpoint_camera: Camera,
            pc: GaussianModel,
            bg_color: torch.Tensor,
            scaling_modifier=1.0,
            render_types: list = None,
            **kwargs,
    ):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            scaling_modifier=scaling_modifier,
            **kwargs,
        )

    def before_training_step(
            self,
            step: int,
            module,
    ):
        return

    def after_training_step(
            self,
            step: int,
            module,
    ):
        return

    def setup(self, stage: str, *args: Any, **kwargs: Any) -> Any:
        pass

    def get_metric_calculators(self) -> Tuple[Union[None, Callable], Union[None, Callable]]:
        return None, None

    def training_setup(self, module: lightning.LightningModule) -> Tuple[
        Optional[Union[
            List[torch.optim.Optimizer],
            torch.optim.Optimizer,
        ]],
        Optional[Union[
            List[torch.optim.lr_scheduler.LRScheduler],
            torch.optim.lr_scheduler.LRScheduler,
        ]]
    ]:
        return None, None

    def on_load_checkpoint(self, module, checkpoint):
        pass

    def get_available_output_types(self) -> Dict:
        return {
            "rgb": "render",
        }

    def is_type_depth_map(self, t: str) -> bool:
        return False

    def is_type_normal_map(self, t: str) -> bool:
        return False
