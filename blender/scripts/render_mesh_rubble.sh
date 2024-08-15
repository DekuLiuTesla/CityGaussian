# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygs2d_rubble_coarse
NAME=citygs2d_rubble
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py --model_path outputs/$NAME --config_path outputs/$COARSE_NAME/config.yaml --voxel_size 0.2 --sdf_trunc 2 --depth_trunc 150


# python tools/generate_traj.py --config outputs/$COARSE_NAME/config.yaml \
#                               --mesh_path "outputs/$NAME/mesh/merged_epoch=19-step=30000" \
#                               --scale_percentile 90 \
#                               --pitch 30

# cd blender
# conda activate blender34

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=1 python render_sun.py \
                     --load_dir "../outputs/$NAME/mesh/merged_epoch=19-step=30000" \
                     --mesh_file rubble_filtered.ply \
                     --config_dir render_cfgs/pixsfm \

# cd ..