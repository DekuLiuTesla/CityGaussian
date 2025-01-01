#!/bin/bash

if [ "${1}" == "" ] || [ "${2}" == "" ]; then
  echo "Usage: get_sam_masks.slurm IMAGE_DIR TRAINED_MODEL_DIR"
  exit 1
fi

srun python utils/get_sam_masks.py "${1}" --preview
srun --gres=gpu:1 --nodes=1 -n1 --ntasks-per-node=1 python utils/get_sam_mask_scales.py "${2}" --preview