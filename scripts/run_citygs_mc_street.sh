# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}


COARSE_NAME=citygs_mc_street_coarse
NAME=block_mc_street_block_all_lr_c36_simple_selection_lr2_light_25_vq
TEST_PATH="data/matrix_city/street/test/block_A_test"
max_block_id=19

# python utils/ply2ckpt.py outputs/$NAME/point_cloud.ply \
#                          -r "outputs/$COARSE_NAME/checkpoints/epoch=8-step=30000.ckpt" \
#                          -s 2

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config configs/$COARSE_NAME.yaml \
    -n $NAME \
    --ckpt_path outputs/$NAME/checkpoints/epoch=8-step=30000.ckpt \
    --model.gaussian.sh_degree 2 \
    --data.path $TEST_PATH \
    --data.params.colmap_block.eval_image_select_mode ratio \
    --data.params.colmap_block.eval_ratio 1.0 \
    --test_speed \
    --save_val \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                    --model_path outputs/$NAME \
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
                                    --ply-path "outputs/$NAME/mesh/epoch=8-step=30000/fuse_post.ply"