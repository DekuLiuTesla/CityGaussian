get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

CONFIG="rubble_c9_r4_lod"
TEST_PATH="data/mill19/rubble-pixsfm/val"

out_name="val_4"  # 4 denotes resolution

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python render_large_lod.py --config config/$CONFIG.yaml --custom_test $TEST_PATH --load_vq

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python metrics_large.py -m output/$CONFIG -t $out_name
