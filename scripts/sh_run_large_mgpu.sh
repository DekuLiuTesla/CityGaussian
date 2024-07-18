# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# large scale dataset, training with multiple GPUs
NAME=rubble_pixsfm_mgpu

# Single GPU at the beginning
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config configs/large_scale.yaml \
    --data.path data/mill19/rubble-pixsfm/train \
    --data.params.colmap.down_sample_factor 4 \
    -n $NAME \
    --logger wandb \
    --project JointGS \
    --max_steps 30000 \

# Then resume, and enable multi-GPU
CUDA_VISIBLE_DEVICES="4,5,6,7" python main.py fit \
    --config configs/large_scale.yaml \
    --trainer configs/ddp.yaml \
    --data.path data/mill19/rubble-pixsfm/train \
    --data.params.colmap.down_sample_factor 4 \
    -n $NAME \
    --logger wandb \
    --project JointGS \
    --max_steps 60000 \
    --ckpt_path last  # find latest checkpoint automatically, or provide a path to checkpoint file

python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.path data/mill19/rubble-pixsfm/val \
    --data.params.colmap.down_sample_factor 4 \
    --data.params.colmap.eval_image_select_mode ratio \
    --data.params.colmap.eval_ratio 1.0 \
    --model.save_val_output true \
