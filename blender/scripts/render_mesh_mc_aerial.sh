# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}


COARSE_NAME=citygs2d_mc_aerial_coarse_lnorm4_wo_vast_sep_depth_init_5
NAME=citygs2d_mc_aerial_lnorm4_wo_vast_sep_depth


# python tools/generate_traj.py --config outputs/$COARSE_NAME/config.yaml \
#                               --mesh_path "outputs/$NAME/mesh/epoch=6-step=30000" \
#                               --data_path data/matrix_city/aerial/test/block_all_test \
#                               --train

# cd blender
# conda activate blender34

# CityGSV2
# load_dir="/home/yang_liu/python_workspace/gaussian-splatting-lightning/outputs/citygs2d_mc_aerial_lnorm4_wo_vast_sep_depth/mesh/epoch=6-step=30000"
# mesh_file="fuse_post.ply"

# GoF
load_dir="/home/yang_liu/python_workspace/gaussian-opacity-fields/outputs/mc_aerial/test/ours_60000/fusion"
mesh_file="mesh_binary_search_7.ply"
name="gof"

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
rm -rf ./output/${name}_mc_aerial
CUDA_VISIBLE_DEVICES=3 python render_sun.py \
                     --load_dir $load_dir \
                     --mesh_file $mesh_file \
                     --save_dir ./output/${name}_mc_aerial \
                     --config_dir render_cfgs/mc

# cd ..