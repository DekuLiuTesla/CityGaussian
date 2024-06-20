# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}


# downsample images
# python utils/image_downsample.py data/urban_scene_3d/sci-art-pixsfm/train/images --factor 4
# python utils/image_downsample.py data/urban_scene_3d/sci-art-pixsfm/val/images --factor 4

NAME=citygs2d_sciart_coarse
DATA_PATH="data/urban_scene_3d/sci-art-pixsfm"

# train and eval coarse model
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config configs/citygs2d_sciart_coarse.yaml \
    -n $NAME \
    --logger wandb \
    --project JointGS \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.path data/urban_scene_3d/sci-art-pixsfm/val \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 1.0 \
    --save_val

NAME=citygs2d_sciart

# generate partition
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/data_partition.py --config_path configs/$NAME.yaml --save_dir $DATA_PATH/train/partition/3_1_3_2D

for num in {0..8}; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting training block '$num'"
            CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python main.py fit --config configs/$NAME.yaml --data.params.colmap_block.block_id $num -n $NAME  --logger wandb --project JointGS & 
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

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py --model_path outputs/citygs2d_sciart_coarse --mesh_res 1024 --unbounded

# merge blocks
python tools/block_merge.py --config_path configs/$NAME.yaml \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$NAME.yaml \
    -n $NAME \
    --ckpt_path outputs/$NAME/checkpoints/merged.ckpt \
    --data.path data/urban_scene_3d/sci-art-pixsfm/val \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 1.0 \
    --save_val \
    --model.correct_color true \