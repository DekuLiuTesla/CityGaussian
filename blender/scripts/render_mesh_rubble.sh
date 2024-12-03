# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

traj_path="/data1/yang_liu/python_workspace/GSPL/data/mill19/rubble-pixsfm/train/traj_ellipse"

declare -a run_args=(
    "/data1/yang_liu/python_workspace/GSPL/outputs/citygs2d_rubble_lnorm4_wo_vast_sep_ssim_depth_trim_v6/mesh/epoch=19-step=30000/fuse_post.ply"
)

declare -a names=(
  "citygs2d_rubble_lnorm4_wo_vast_sep_ssim_depth_trim_v6"
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
                     --save_dir ./output/${NAME}_rubble \
                     --config_dir render_cfgs/pixsfm \
                     --fps 30 \
                     --write_cover \
                    #  --image_only \

        break
      else
        echo "No GPU available at the moment. Retrying in 2 minute."
        sleep 120
      fi
  done
done