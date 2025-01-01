from dataclasses import dataclass
import lightning
import torch
from typing import Any, Union, List, Tuple, Optional, Dict, Callable
from internal.configs.instantiate_config import InstantiatableConfig
from internal.cameras.cameras import Camera
from internal.models.gaussian import GaussianModel


class RendererOutputTypes:
    RGB: int = 1
    GRAY: int = 2
    NORMAL_MAP: int = 3
    FEATURE_MAP: int = 4
    OTHER: int = 65535  # must provide a visualizer


RendererOutputVisualizer = Callable[[torch.Tensor, Dict, "RendererOutputInfo"], torch.Tensor]


@dataclass
class RendererOutputInfo:
    key: str
    """The key used to retrieve value from the dictionary returned by `forward()`"""

    type: int = RendererOutputTypes.RGB
    """One defined in `RendererOutputTypes` above"""

    visualizer: RendererOutputVisualizer = None
    """
    The first parameter is the value retrieved from the dict returned by `forward()`. 
    The second parameter is the dict returned by `forward()`. 
    The Third one is a `RendererOutputInfo` instance.
    """

    def __post_init__(self):
        if self.type == RendererOutputTypes.OTHER and self.visualizer is None:
            raise ValueError("Visualizer must be provided when `type` is `OTHER`")

        # TODO: set visualizer automatically if it is None


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
            render_types: list = None,
            **kwargs,
    ):
        return self(
            viewpoint_camera=viewpoint_camera,
            pc=pc,
            bg_color=bg_color,
            render_types=render_types,
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

    def setup_web_viewer_tabs(self, viewer, server, tabs):
        pass

    def get_available_outputs(self) -> Dict[str, RendererOutputInfo]:
        return {
            "rgb": RendererOutputInfo("render")
        }


@dataclass
class RendererConfig(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> Renderer:
        raise NotImplementedError()
