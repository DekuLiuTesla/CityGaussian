# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# large scale dataset
NAME=canteen1

python utils/image_downsample.py data/dji_test/beijing/canteen1/images --factor 2

# Single GPU at the beginning
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config configs/custom_gsplat.yaml \
    --data.path data/dji_test/beijing/canteen1 \
    --trainer.check_val_every_n_epoch 50 \
    --data.params.colmap.down_sample_factor 2 \
    -n $NAME \
    --logger wandb \
    --project JointGS \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.params.colmap.down_sample_factor 2 \
    --model.save_val_output true \