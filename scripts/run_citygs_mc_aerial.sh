# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygs_mc_aerial_coarse
NAME=citygs_mc_aerial_light_56_vq
TEST_PATH=data/matrix_city/aerial/test/block_all_test
max_block_id=35

# python utils/ply2ckpt.py outputs/$NAME/point_cloud.ply \
#                          -r "outputs/$COARSE_NAME/checkpoints/epoch=6-step=30000.ckpt" \
#                          -s 2

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$NAME.yaml \
    -n $NAME \
    --data.path $TEST_PATH \
    --data.params.estimated_depth_colmap_block.eval_image_select_mode ratio \
    --data.params.estimated_depth_colmap_block.eval_ratio 1.0 \
    --test_speed \
    --save_val \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$NAME \
                                    --config_path outputs/$COARSE_NAME/config.yaml \
                                    --voxel_size 0.01 \
                                    --sdf_trunc 0.04 \
                                    --depth_trunc 5.0 \
                                    --use_trim_renderer \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_mc.py \
                                    --scene Block_all_ds \
                                    --dataset-dir data/matrix_city/point_cloud_ds20/aerial \
                                    --ply-path "outputs/$NAME/mesh/epoch=6-step=30000/fuse_post.ply"

