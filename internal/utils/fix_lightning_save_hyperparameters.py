"""
https://github.com/Lightning-AI/pytorch-lightning/pull/18105
"""

from contextlib import contextmanager
import lightning.pytorch.core.mixins.hparams_mixin

if hasattr(lightning.pytorch.core.mixins.hparams_mixin, "_given_hyperparameters_context"):
    @contextmanager
    def fix_save_hyperparameters(*args, **kwargs):
        yield

    lightning.pytorch.core.mixins.hparams_mixin._given_hyperparameters_context = fix_save_hyperparameters
