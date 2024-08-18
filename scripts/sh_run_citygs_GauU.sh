# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

SCENE=LFLS
COARSE_NAME=citygs_flatten_lfls_coarse
NAME=citygs_flatten_lfls
DATA_PATH=data/GauU_Scene/$SCENE
max_block_id=8

# downsample images
# python utils/image_downsample.py data/GauU_Scene/LFLS/images --factor 3.4175

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
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$COARSE_NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 0.01 \
                                    --sdf_trunc 0.1 \
                                    --depth_trunc 10
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$COARSE_NAME/config.yaml \
    --save_val \

# generate partition
python tools/data_partition.py --config_path configs/$NAME.yaml

for num in $(seq 0 $max_block_id); do
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
            echo "No GPU available at the moment. Retrying in 3 minute."
            sleep 120
        fi
    done
done
wait

merge blocks
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/block_merge.py --config_path configs/$NAME.yaml \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$NAME.yaml \
    -n $NAME \
    --save_val \
    # --model.correct_color true \


gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 0.01 \
                                    --sdf_trunc 0.1 \
                                    --depth_trunc 10

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_gauu.py \
                                    --scene lidar \
                                    --dataset-dir data/GauU_Scene/Downsampled/$SCENE \
                                    --transform-path data/GauU_Scene/$SCENE/transformation.txt \
                                    --ply-path "outputs/$NAME/mesh/merged_epoch=32-step=30000/fuse_post.ply"