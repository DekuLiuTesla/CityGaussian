# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygs2d_mc_aerial_coarse_lnorm4_wo_vast_sep_depth_init_5
NAME=citygs2d_mc_aerial_lnorm4_wo_vast_sep_depth

declare -a depth_truncs=(3.0 3.5 4.0)

for depth_trunc in "${depth_truncs[@]}"; do
  while true; do
    gpu_id=$(get_available_gpu)
    if [[ -n $gpu_id ]]; then
      echo "GPU $gpu_id is available. Starting mesh extraction with depth truncation of $depth_trunc"

      CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                          --model_path outputs/$NAME \
                                          --config_path outputs/$COARSE_NAME/config.yaml \
                                          --voxel_size 0.01 \
                                          --sdf_trunc 0.04 \
                                          --mesh_name fuse_dtrunc_$depth_trunc \
                                          --depth_trunc $depth_trunc # 2.0 for metrics

      # Increment the port number for the next run
      ((port++))
      # Allow some time for the process to initialize and potentially use GPU memory
      sleep 60
      break
    else
      echo "No GPU available at the moment. Retrying in 1 minute."
      sleep 60
    fi
  done
done