from typing import Literal, Optional, Any
from dataclasses import dataclass, field
from internal.configs.instantiate_config import InstantiatableConfig
from torch.optim import Optimizer, lr_scheduler
from torch.optim.lr_scheduler import LRScheduler
import numpy as np


# `ExponentialDecayScheduler` is copied from NeRFStudio

class Scheduler(InstantiatableConfig):
    def instantiate(self, *args, **kwargs) -> "SchedulerImpl":
        raise NotImplementedError()


class SchedulerImpl:
    """Base scheduler"""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        """Abstract method that returns a scheduler object.

        Args:
            optimizer: The optimizer to use.
            lr_init: The initial learning rate.
        Returns:
            The scheduler object.
        """
        raise NotImplementedError()


@dataclass
class ExponentialDecayScheduler(Scheduler):
    """Config for exponential decay scheduler with warmup"""

    """target class to instantiate"""
    lr_pre_warmup: float = 1e-8
    """Learning rate before warmup."""
    lr_final: Optional[float] = None
    """Final learning rate. If not provided, it will be set to the optimizers learning rate."""
    warmup_steps: int = 0
    """Number of warmup steps."""
    max_steps: int = 100000
    """The maximum number of steps."""
    ramp: Literal["linear", "cosine"] = "cosine"
    """The ramp function to use during the warmup."""

    def instantiate(self, *args, **kwargs) -> Any:
        return ExponentialDecaySchedulerImpl(self)


class ExponentialDecaySchedulerImpl(SchedulerImpl):
    """Exponential decay scheduler with linear warmup. Scheduler first ramps up to `lr_init` in `warmup_steps`
    steps, then exponentially decays to `lr_final` in `max_steps` steps.
    """

    config: ExponentialDecayScheduler

    def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> LRScheduler:
        if self.config.lr_final is None:
            lr_final = lr_init
        else:
            lr_final = self.config.lr_final

        def func(step):
            if step < self.config.warmup_steps:
                if self.config.ramp == "cosine":
                    lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
                        0.5 * np.pi * np.clip(step / self.config.warmup_steps, 0, 1)
                    )
                else:
                    lr = (
                            self.config.lr_pre_warmup
                            + (lr_init - self.config.lr_pre_warmup) * step / self.config.warmup_steps
                    )
            else:
                t = np.clip(
                    (step - self.config.warmup_steps) / (self.config.max_steps - self.config.warmup_steps), 0, 1
                )
                lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
            return lr / lr_init  # divided by lr_init because the multiplier is with the initial learning rate

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
        return scheduler
