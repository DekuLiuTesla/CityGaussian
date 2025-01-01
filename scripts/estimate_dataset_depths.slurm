#!/bin/bash

# 4 nodes, 2 GPUs per node: sbatch --nodes=4 --gres=gpu:2 --ntasks-per-node=2 scripts/estimate_dataset_depths.slurm DATASET_DIR --preview

if [ "${1}" == "" ]; then
  echo "dataset directory is required"
  exit 1
fi

IMAGE_DIR="${1}/images"

srun python utils/run_depth_anything_v2.py "${IMAGE_DIR}" "${@:2}"
srun --gres=gpu:1 --nodes=1 -n1 --ntasks-per-node=1 python utils/get_depth_scales.py "${1}"