# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygs2d_building_coarse_lnorm4_wo_vast_sep_ssim_depth_init_5_v6
NAME=citygs2d_building_lnorm4_wo_vast_sep_ssim_depth_trim_v6
TEST_PATH=data/mill19/building-pixsfm/val
max_block_id=19

# ============================================= downsample images =============================================
# python utils/image_downsample.py data/mill19/building-pixsfm/train/images --factor 4
# python utils/image_downsample.py data/mill19/building-pixsfm/val/images --factor 4

# ===================================== generate depth with depth-anything-V2 =================================
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/estimate_dataset_depths.py \
#                                     data/mill19/building-pixsfm/train \
#                                     -d 4 \

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/estimate_dataset_depths.py \
#                                     $TEST_PATH \
#                                     -d 4 \

# ============================================= train&eval coarse model =============================================
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                                    --config configs/$COARSE_NAME.yaml \
                                    -n $COARSE_NAME \
                                    --logger wandb \
                                    --project JointGS \


gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$COARSE_NAME.yaml \
    -n $COARSE_NAME \
    --data.path $TEST_PATH \
    --data.params.estimated_depth_colmap_block.eval_image_select_mode ratio \
    --data.params.estimated_depth_colmap_block.eval_ratio 1.0 \
    --save_val \
    --model.correct_color true \

# ============================================= generate partition =============================================
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/data_partition.py --config_path configs/$NAME.yaml

# ============================================= train&eval tuned model =============================================
for num in $(seq 0 $max_block_id); do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting training block '$num'"
            CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python main.py fit \
                --config configs/$NAME.yaml \
                --data.params.estimated_depth_colmap_block.block_id $num \
                -n $NAME \
                --logger wandb \
                --project JointGS & 
            # Increment the port number for the next run
            ((port++))
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done
wait

# merge blocks
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/block_merge.py --config_path configs/$NAME.yaml \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$COARSE_NAME/config.yaml \
    -n $NAME \
    --data.path $TEST_PATH \
    --data.params.estimated_depth_colmap_block.eval_image_select_mode ratio \
    --data.params.estimated_depth_colmap_block.eval_ratio 1.0 \
    --save_val \
    --model.correct_color true \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 0.25 \
                                    --sdf_trunc 1.5 \
                                    --depth_trunc 250 \

# ============================================= remove block results (if you find result OK) =============================================
# for num in $(seq 0 $max_block_id); do
#     rm -rf outputs/$NAME/blocks/block_${num}/checkpoints
#     echo "Removed checkpoints for block $num"
# done

# ============================================= vector quantization =============================================
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python tools/vectree_lightning.py \
#                                     --coarse_config outputs/$COARSE_NAME/config.yaml \
#                                     --input_path outputs/$NAME/checkpoints/epoch=32-step=30000.ckpt \
#                                     --save_path outputs/$NAME/vectree \
#                                     --sh_degree 2 \
#                                     --skip_quantize \
#                                     --no_save_ply \
