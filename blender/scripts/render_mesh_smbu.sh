# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=5000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

traj_path="/data1/yang_liu/python_workspace/GSPL/data/GauU_Scene/SMBU/traj_ellipse"

declare -a run_args=(
    "/data1/yang_liu/python_workspace/GSPL/outputs/citygs_smbu/mesh/epoch=60-step=30000/fuse_dtrunc_2.5_post.ply"
    "/data1/yang_liu/python_workspace/GSPL/outputs/citygs2d_smbu_coarse_lnorm4_wo_vast_6w/mesh/epoch=120-step=60000/fuse_dtrunc_2.5_post.ply"
    "/data1/yang_liu/python_workspace/GSPL/outputs/citygs2d_smbu_lnorm4_wo_vast_sep_ssim_depth_trim_v6/mesh/epoch=60-step=30000/fuse_vox2_dtrunc2.5_post.ply"
    "/data1/yang_liu/python_workspace/gaussian-opacity-fields/outputs/SMBU/test/ours_30000/fusion/mesh_binary_search_7.ply"
    "/data1/yang_liu/python_workspace/SuGaR/output/refined_mesh/SMBU/sugarfine_3Dgs14000_sdfestim02_sdfnorm02_level03_decim2000000_normalconsistency01_gaussperface1.obj"
)

declare -a names=(
  "citygs_smbu"
  "citygs2d_smbu_coarse_lnorm4_wo_vast_6w"
  "citygs2d_smbu_lnorm4_wo_vast_sep_ssim_depth_trim"
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
                     --save_dir ./output/${NAME}_smbu \
                     --config_dir render_cfgs/gauu \
                     --write_cover \
                     --image_only \

        break
      else
        echo "No GPU available at the moment. Retrying in 2 minute."
        sleep 120
      fi
  done
done

# cd ..