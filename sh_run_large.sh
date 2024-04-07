# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# large scale dataset
NAME=rubble_pixsfm

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

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/lightning_logs/version_0/config.yaml \
    --data.path data/mill19/rubble-pixsfm/val \
    --data.params.colmap.down_sample_factor 4 \
    --data.params.colmap.eval_image_select_mode ratio \
    --data.params.colmap.eval_ratio 1.0 \
    --model.save_val_output true \
