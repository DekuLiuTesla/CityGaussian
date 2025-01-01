#!/bin/bash

PARTITION_DATA_PATH=~/dataset/Mega-NeRF/rubble-pixsfm/colmap/dense_max_1600/partitions-size_2.9-enlarge_0.1-visibility_0.9_0.25
PROJECT_NAME=MegaNeRF-rubble-view_independent-hard_depth

# NOTES
# * `srun` is invoked by the python script
# * these options are vital for the correctness of job scheduling
#   * --gpus=1
#   * --nodes=1
#   * --ntasks=1
#   * --exclusive
#
# submitting: sbatch --gpus=6 --cpus-per-gpu=32 scripts/train-meganerf_rubble-partitions.slurm
#  remember to specify the value of `--cpus-per-gpu=`, or only one cpu core will be assigned to each process even with the `DefCpuPerGPU` set
python3 utils/train_colmap_partitions_v2.py \
  ${PARTITION_DATA_PATH} \
  -p ${PROJECT_NAME} \
  --scalable-config utils/scalable_param_configs/appearance-with_scheduler-depth_reg.yaml \
  --config configs/appearance_embedding_renderer/view_independent-lr_0.005-with_scheduler-estimated_depth_reg-hard_depth.yaml \
  -- \
  --data.parser.appearance_groups appearance_image_dedicated \
  --model.gaussian.optimization.spatial_lr_scale 1.5 \
  -- \
  --gpus=1 \
  --nodes=1 \
  --ntasks=1 \
  --exclusive