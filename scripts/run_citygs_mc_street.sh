# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}


COARSE_NAME=citygs_mc_street_coarse_cull_1_scaled
NAME=citygs_mc_street
TEST_PATH="data/matrix_city/street/test/block_A_test"
max_block_id=19

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/estimate_dataset_depths.py \
#                                     data/matrix_city/street/train/block_A \

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/estimate_dataset_depths.py \
#                                     $TEST_PATH \

# train and eval coarse model
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
    --config outputs/$COARSE_NAME/config.yaml \
    --data.path $TEST_PATH \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 1.0 \
    --save_val

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$COARSE_NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 1 \
                                    --sdf_trunc 4 \
                                    --depth_trunc 500 \
                                    --use_trim_renderer \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_mc.py \
                                    --scene Block_A_ds \
                                    --dataset-dir data/matrix_city/point_cloud_ds20/street \
                                    --ply-path "outputs/$COARSE_NAME/mesh/epoch=8-step=30000/fuse_post.ply"


# generate partition
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/data_partition.py --config_path configs/$NAME.yaml

for num in $(seq 0 $max_block_id); do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting training block '$num'"
            CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python main.py fit \
                --config configs/$NAME.yaml \
                --data.params.colmap_block.block_id $num \
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
python tools/block_merge.py --config_path configs/$NAME.yaml \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$NAME.yaml \
    -n $NAME \
    --ckpt_path outputs/$NAME/checkpoints/merged.ckpt \
    --data.path $TEST_PATH \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 1.0 \
    --save_val \
#     # --model.correct_color true \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 1 \
                                    --sdf_trunc 4 \
                                    --depth_trunc 500 \