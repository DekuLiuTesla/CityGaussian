# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygs_upper_coarse
NAME=citygs_upper_light_55_vq
max_block_id=8

# python utils/ply2ckpt.py outputs/$NAME/point_cloud.ply \
#                          -r "outputs/$COARSE_NAME/checkpoints/epoch=48-step=30000.ckpt" \
#                          -s 2

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$NAME.yaml \
    --data.params.colmap_block.split_mode experiment \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 0.1 \
    -n $NAME \
    --save_val \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 0.01 \
                                    --sdf_trunc 0.04 \
                                    --depth_trunc 2.0 \
                                    --use_trim_renderer
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_gauu.py \
                                    --scene CUHK_UPPER_COLMAP_ds_35 \
                                    --dataset-dir data/GauU_Scene/CUHK_UPPER_COLMAP \
                                    --transform-path data/GauU_Scene/Downsampled/CUHK_UPPER/transform.txt \
                                    --ply-path "outputs/$NAME/mesh/epoch=48-step=30000/fuse_post.ply"

# for num in $(seq 0 $max_block_id); do
#     rm -rf outputs/$NAME/blocks/block_${num}/checkpoints
#     echo "Removed checkpoints for block $num"
# done