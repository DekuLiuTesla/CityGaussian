import os
import torch
from tqdm.auto import tqdm
from internal.models.vanilla_gaussian import VanillaGaussianModel
from internal.utils.partitioning_utils import MinMaxBoundingBox
from internal.utils.gaussian_model_loader import GaussianModelLoader
from train_partitions import PartitionTrainingConfig, PartitionTraining


def get_trained_partitions(
        partition_dir: str,
        project_name: str,
        min_images: int,
        n_processes: int = 1,
        process_id: int = 1,
):
    partition_training = PartitionTraining(PartitionTrainingConfig(
        partition_dir=partition_dir,
        project_name=project_name,
        min_images=min_images,
        n_processes=n_processes,
        process_id=process_id,
        dry_run=False,
        extra_epoches=0,
    ))
    trainable_partition_idx_list = partition_training.get_trainable_partition_idx_list(
        min_images=min_images,
        n_processes=n_processes,
        process_id=process_id,
    )

    partition_bounding_boxes = partition_training.partition_coordinates.get_bounding_boxes(partition_training.scene["scene_config"]["partition_size"])

    # ensure that all required partitions exist
    trained_partitions = []
    for partition_idx in tqdm(trainable_partition_idx_list, desc="Searching checkpoints"):
        partition_id_str = partition_training.get_partition_id_str(partition_idx)
        model_dir = os.path.join(partition_training.project_output_dir, partition_id_str)
        ckpt_file = GaussianModelLoader.search_load_file(model_dir)
        assert ckpt_file.endswith(".ckpt"), "checkpoint of partition #{} ({}) can not be found in {}".format(partition_idx, partition_id_str, partition_training.project_output_dir)
        trained_partitions.append((
            partition_idx,
            partition_id_str,
            ckpt_file,
            partition_bounding_boxes[partition_idx],
        ))

    orientation_transformation = partition_training.scene["extra_data"]["rotation_transform"] if partition_training.scene["extra_data"] is not None else None

    return partition_training, trained_partitions, orientation_transformation


def get_partition_gaussian_mask(
        means: torch.Tensor,
        partition_bounding_box: MinMaxBoundingBox,
        orientation_transform: torch.Tensor = None,
):
    if orientation_transform is not None:
        means = means @ orientation_transform[:3, :3].T

    # include min bound, exclude max bound
    is_ge_min = torch.prod(torch.ge(means[..., :2], partition_bounding_box.min), dim=-1)
    is_lt_max = torch.prod(torch.lt(means[..., :2], partition_bounding_box.max), dim=-1)
    is_in_bounding_box = torch.logical_and(is_ge_min, is_lt_max)

    return is_in_bounding_box


def split_partition_gaussians(ckpt: dict, partition_bounding_box: MinMaxBoundingBox, orientation_transform: torch.Tensor = None) -> tuple[
    VanillaGaussianModel,
    dict[str, torch.Tensor],
    torch.Tensor,
]:
    model = GaussianModelLoader.initialize_model_from_checkpoint(ckpt, "cpu")
    is_in_partition = get_partition_gaussian_mask(model.get_means(), partition_bounding_box, orientation_transform=orientation_transform)

    inside_part = {}
    outside_part = {}
    for k, v in model.properties.items():
        inside_part[k] = v[is_in_partition]
        outside_part[k] = v[~is_in_partition]

    model.properties = inside_part

    return model, outside_part, is_in_partition
