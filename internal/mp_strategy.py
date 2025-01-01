from typing import Optional, List, Union, Any, Dict

import torch
import torch.distributed
import lightning.pytorch as pl
from lightning.fabric.plugins import ClusterEnvironment, CheckpointIO
from lightning.fabric.utilities.types import ReduceOp, _PATH
from lightning.pytorch.plugins import PrecisionPlugin
from lightning.pytorch.strategies.parallel import ParallelStrategy
from lightning.pytorch.strategies.strategy import TBroadcast
from torch import Tensor
from lightning.fabric.utilities.seed import reset_seed
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only

from lightning.fabric.utilities.distributed import (
    _get_default_process_group_backend_for_device,
    _init_dist_connection,
    _sync_ddp_if_available,
)
try:
    from lightning.fabric.utilities.distributed import _distributed_available
except ImportError:
    from lightning.fabric.utilities.distributed import _distributed_is_initialized as _distributed_available

from lightning.fabric.utilities.distributed import group as _group
from lightning.pytorch.strategies.launchers.subprocess_script import _SubprocessScriptLauncher


class MPStrategy(ParallelStrategy):
    def __init__(
            self,
            accelerator: Optional["pl.accelerators.Accelerator"] = None,
            parallel_devices: Optional[List[torch.device]] = None,
            cluster_environment: Optional[ClusterEnvironment] = None,
            checkpoint_io: Optional[CheckpointIO] = None,
            precision_plugin: Optional[PrecisionPlugin] = None,
            process_group_backend: Optional[str] = None,
    ):
        super().__init__(accelerator, parallel_devices, cluster_environment, checkpoint_io, precision_plugin)

        self._num_nodes = 1
        self._process_group_backend = process_group_backend

    def setup(self, trainer: "pl.Trainer") -> None:
        self.model_to_device()
        super().setup(trainer)

    def setup_environment(self) -> None:
        self.setup_distributed()
        super().setup_environment()

    def setup_distributed(self) -> None:
        reset_seed()
        self.set_world_ranks()
        self._process_group_backend = self._get_process_group_backend()
        assert self.cluster_environment is not None
        _init_dist_connection(self.cluster_environment, self._process_group_backend)

    def _get_process_group_backend(self) -> str:
        return self._process_group_backend or _get_default_process_group_backend_for_device(self.root_device)

    def set_world_ranks(self) -> None:
        if self.cluster_environment is not None:
            self.cluster_environment.set_global_rank(self.node_rank * self.num_processes + self.local_rank)
            self.cluster_environment.set_world_size(self.num_nodes * self.num_processes)
        # `LightningEnvironment.set_global_rank` will do this too, but we cannot rely on that implementation detail
        # additionally, for some implementations, the setter is a no-op, so it's safer to access the getter
        rank_zero_only.rank = self.global_rank

    def _configure_launcher(self) -> None:
        assert self.cluster_environment is not None
        if not self.cluster_environment.creates_processes_externally:
            self._launcher = _SubprocessScriptLauncher(self.cluster_environment, self.num_processes, self.num_nodes)

    @property
    def num_processes(self) -> int:
        return len(self.parallel_devices) if self.parallel_devices is not None else 0

    @property
    def num_nodes(self) -> int:
        return self._num_nodes

    @num_nodes.setter
    def num_nodes(self, num_nodes: int) -> None:
        # note that world ranks is related to num_nodes, when resetting it, need to reset world ranks
        self._num_nodes = num_nodes

    @property
    def process_group_backend(self) -> Optional[str]:
        return self._process_group_backend

    @property
    def root_device(self) -> torch.device:
        assert self.parallel_devices is not None
        return self.parallel_devices[self.local_rank]

    def model_to_device(self) -> None:
        assert self.model is not None
        self.model.to(self.root_device)

    def reduce(self, tensor: Union[Tensor, Any], group: Optional[Any] = None, reduce_op: Optional[Union[ReduceOp, str]] = "mean") -> Union[Tensor, Any]:
        if isinstance(tensor, Tensor):
            return _sync_ddp_if_available(tensor, group, reduce_op=reduce_op)
        return tensor

    def determine_ddp_device_ids(self) -> Optional[List[int]]:
        if self.root_device.type == "cpu":
            return None
        return [self.root_device.index]

    def barrier(self, name: Optional[str] = None) -> None:
        if not _distributed_available():
            return
        if torch.distributed.get_backend() == "nccl":
            torch.distributed.barrier(device_ids=self.determine_ddp_device_ids())
        else:
            torch.distributed.barrier()

    def broadcast(self, obj: TBroadcast, src: int = 0) -> TBroadcast:
        if not _distributed_available():
            return obj
        obj = [obj]
        if self.global_rank != src:
            obj = [None]  # type: ignore[list-item]
        torch.distributed.broadcast_object_list(obj, src, group=_group.WORLD)
        return obj[0]

    def save_checkpoint(self, checkpoint: Dict[str, Any], filepath: _PATH, storage_options: Optional[Any] = None) -> None:
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options=storage_options)
