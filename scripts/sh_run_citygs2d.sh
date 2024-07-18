# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}


# downsample images
# python utils/image_downsample.py data/mill19/rubble-pixsfm/train/images --factor 4
# python utils/image_downsample.py data/mill19/rubble-pixsfm/val/images --factor 4

COARSE_NAME=citygs2d_rubble_coarse
DATA_PATH="data/mill19/rubble-pixsfm"

# train and eval coarse model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config configs/$COARSE_NAME.yaml \
    -n debug_$COARSE_NAME \
    --logger wandb \
    --project JointGS \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$COARSE_NAME/config.yaml \
    --data.path data/mill19/rubble-pixsfm/val \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 1.0 \
    --save_val

# NAME=citygs_rubble

# # generate partition
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python tools/data_partition.py --config_path configs/$NAME.yaml

# for num in {0..8}; do
#     while true; do
#         gpu_id=$(get_available_gpu)
#         if [[ -n $gpu_id ]]; then
#             echo "GPU $gpu_id is available. Starting training block '$num'"
#             CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python main.py fit \
#                 --config configs/$NAME.yaml \
#                 --data.params.colmap_block.block_id $num \
#                 -n $NAME \
#                 --logger wandb \
#                 --project JointGS & 
#             # Increment the port number for the next run
#             ((port++))
#             # Allow some time for the process to initialize and potentially use GPU memory
#             sleep 120
#             break
#         else
#             echo "No GPU available at the moment. Retrying in 2 minute."
#             sleep 120
#         fi
#     done
# done
# wait

# # merge blocks
# python tools/block_merge.py --config_path configs/$NAME.yaml \

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
#     --config configs/$NAME.yaml \
#     -n $NAME \
#     --ckpt_path outputs/$NAME/checkpoints/merged.ckpt \
#     --data.path data/mill19/rubble-pixsfm/val \
#     --data.params.colmap_block.eval_image_select_mode ratio \
#     --data.params.colmap_block.eval_ratio 1.0 \
#     --save_val \
    # --model.correct_color true \

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py --model_path outputs/$NAME --config_path outputs/$COARSE_NAME/config.yaml --voxel_size 0.4 --sdf_trunc 2 --depth_trunc 300
