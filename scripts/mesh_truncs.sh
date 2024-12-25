# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=7000
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygs2d_upper_coarse_lnorm4_wo_vast_sep_depth_init_5_r0_v6
NAME=citygs2d_upper_coarse_lnorm4_wo_vast_sep_depth_init_5_r0_v6

declare -a sdf_truncs=(0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075)

for sdf_trunc in "${sdf_truncs[@]}"; do
  depth_trunc=5.0
  while true; do
    gpu_id=$(get_available_gpu)
    if [[ -n $gpu_id ]]; then
      echo "GPU $gpu_id is available. Starting mesh extraction with depth truncation of $depth_trunc and SDF truncation of $sdf_trunc."

      CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py \
                                          --model_path outputs/$NAME \
                                          --config_path outputs/$COARSE_NAME/config.yaml \
                                          --voxel_size 0.01 \
                                          --sdf_trunc $sdf_trunc \
                                          --depth_trunc $depth_trunc \
                                          --mesh_name fuse_sdftrunc_${sdf_trunc}_dtrunc_$depth_trunc & \

      # CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_mc.py \
      #                               --scene Block_all_ds \
      #                               --dataset-dir data/matrix_city/point_cloud_ds20/aerial \
      #                               --ply-path "outputs/$NAME/mesh/epoch=48-step=30000/fuse_sdftrunc_${sdf_trunc}_dtrunc_${depth_trunc}_post.ply" > "outputs/logs/fuse_sdftrunc_${sdf_trunc}_dtrunc_${depth_trunc}_post.log"  &

      # CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_gauu.py \
      #                               --scene CUHK_UPPER_COLMAP_ds_35 \
      #                               --dataset-dir data/GauU_Scene/CUHK_UPPER_COLMAP \
      #                               --transform-path data/GauU_Scene/Downsampled/CUHK_UPPER/transform.txt \
      #                               --ply-path "outputs/$COARSE_NAME/mesh/epoch=48-step=30000/fuse_sdftrunc_${sdf_trunc}_dtrunc_${depth_trunc}_post.ply" &

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
