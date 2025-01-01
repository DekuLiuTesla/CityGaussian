#!/bin/bash

# 4 nodes, 2 GPUs per node: sbatch --nodes=4 --gres=gpu:2 --ntasks-per-node=2 scripts/sd_feature_extraction.slurm IMAGE_DIR

if [ "${1}" == "" ]; then
  echo "image directory is required"
  exit 1
fi

srun python utils/sd_feature_extraction.py "${1}" "${@:2}"