get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

NAME=citygs2d_rubble_2grad
BLOCK_ID=0

# train and eval coarse model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id nsys profile -o output --cuda-memory-usage true --force-overwrite true python main.py fit \
    --config configs/$NAME.yaml \
    --ckpt_path outputs/$NAME/blocks/block_$BLOCK_ID/checkpoints/"epoch=42-step=6999.ckpt" \
    --data.params.colmap_block.block_id $BLOCK_ID \
    -n $NAME \