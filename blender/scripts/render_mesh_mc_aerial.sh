# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

traj_path="/data1/yang_liu/python_workspace/GSPL/data/matrix_city/aerial/test/block_all_test/traj"

declare -a run_args=(
    "/data1/yang_liu/python_workspace/GSPL/outputs/mc_aerial_c36/mesh/point_cloud/fuse_post.ply"
    "/data1/yang_liu/python_workspace/GSPL/outputs/citygs2d_mc_aerial_coarse_lnorm4_wo_vast_6w/mesh/epoch=11-step=60000/fuse_post.ply"
    "/data1/yang_liu/python_workspace/GSPL/outputs/citygs2d_mc_aerial_lnorm4_wo_vast_sep_depth_r0/mesh/epoch=6-step=30000/fuse_post.ply"
    "/data1/yang_liu/python_workspace/SuGaR/output/refined_mesh/block_all/sugarfine_3Dgs14000_sdfestim02_sdfnorm02_level03_decim8000000_normalconsistency01_gaussperface1.obj"
    "/data1/yang_liu/python_workspace/gaussian-opacity-fields/outputs/mc_aerial/test/ours_60000/fusion/mesh_binary_search_7.ply"
)

declare -a names=(
  "mc_aerial_c36"
  "citygs2d_mc_aerial_coarse_lnorm4_wo_vast_6w"
  "citygs2d_mc_aerial_lnorm4_wo_vast_sep_depth_r0_test"
  "sugarfine_3Dgs14000_sdfestim02_sdfnorm02_level03_decim8000000_normalconsistency01_gaussperface1"
  "gof_mesh_binary_search_7"
)

for i in "${!run_args[@]}"; do
  load_path=${run_args[$i]}
  NAME=${names[$i]}
  while true; do
      gpu_id=$(get_available_gpu)
      if [[ -n $gpu_id ]]; then
        echo "GPU $gpu_id is available."

        CUDA_VISIBLE_DEVICES=gpu_id python render_sun.py \
                     --load_path $load_path \
                     --traj_path $traj_path \
                     --save_dir ./output/${NAME}_mc_aerial \
                     --config_dir render_cfgs/mc \
                     --write_cover \
                     --image_only

        break
      else
        echo "No GPU available at the moment. Retrying in 2 minute."
        sleep 120
      fi
  done
done

# cd ..