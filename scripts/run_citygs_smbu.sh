# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

SCENE=SMBU
COARSE_NAME=citygs2d_smbu_coarse_lnorm4_wo_vast_no_elo
NAME=citygs2d_smbu_lnorm4_wo_vast_no_elo
DATA_PATH=data/GauU_Scene/$SCENE
max_block_id=8

# downsample images
# python utils/image_downsample.py data/GauU_Scene/SMBU/images --factor 3.4175

# python utils/ckpt2ply.py outputs/citygs_smbu

# conduct compression

# python utils/ply2ckpt.py outputs/$NAME/point_cloud.ply \
#                          -r "outputs/$COARSE_NAME/checkpoints/epoch=60-step=30000.ckpt" \
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
    --test_speed

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
                                    --scene SMBU_ds_35 \
                                    --dataset-dir data/GauU_Scene/SMBU \
                                    --transform-path data/GauU_Scene/Downsampled/SMBU/transform.txt \
                                    --ply-path "outputs/$NAME/mesh/epoch=60-step=30000/fuse_post.ply"

# for num in $(seq 0 $max_block_id); do
#     rm -rf outputs/$NAME/blocks/block_${num}/checkpoints
#     echo "Removed checkpoints for block $num"
# done