# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

traj_path="/home/yang_liu/python_workspace/gaussian-splatting-lightning/data/matrix_city/street/test/block_A_test/traj"

declare -a run_args=(
    "/home/yang_liu/python_workspace/gaussian-splatting-lightning/outputs/block_mc_street_block_all_lr_c36_simple_selection_lr2/mesh/point_cloud/fuse_post.ply"
    "/home/yang_liu/python_workspace/gaussian-splatting-lightning/outputs/citygs2d_mc_street_coarse_lnorm4_wo_vast_6w/mesh/epoch=15-step=60000/fuse_post.ply"
    "/home/yang_liu/python_workspace/gaussian-splatting-lightning/outputs/citygs2d_mc_street_lnorm4_wo_vast_sep_depth_rgb_only_norm_ubd1e1_no_trim/mesh/epoch=8-step=30000/fuse_post.ply"
    "/home/yang_liu/python_workspace/gaussian-opacity-fields/outputs/mc_street/test/ours_60000/fusion/mesh_binary_search_7.ply"
    "/home/yang_liu/python_workspace/SuGaR/output/refined_mesh/block_A/sugarfine_3Dgs14000_sdfestim02_sdfnorm02_level03_decim2000000_normalconsistency01_gaussperface1.obj"
)

declare -a names=(
  "block_mc_street_block_all_lr_c36_simple_selection_lr2"
  "citygs2d_mc_street_coarse_lnorm4_wo_vast_6w"
  "citygs2d_mc_street_lnorm4_wo_vast_sep_depth_rgb_only_norm_ubd1e1_no_trim"
  "gof_mesh_binary_search_7"
  "sugarfine_3Dgs14000_sdfestim02_sdfnorm02_level03_decim2000000_normalconsistency01_gaussperface1"
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
                     --save_dir ./output/${NAME}_mc_street \
                     --config_dir render_cfgs/mc_street \
                     --image_only \
                     --write_cover &
        
        sleep 120
        
        break
      else
        echo "No GPU available at the moment. Retrying in 2 minute."
        sleep 120
      fi
  done
done

# cd ..