get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

CONFIG="block_rubble_all_lr_c9_loss_12_r4_lod"
TEST_PATH="data/mill19/rubble-pixsfm/val"

CUDA_VISIBLE_DEVICES=$(get_available_gpu) python render_large_lod.py --config config/$CONFIG.yaml --custom_test $TEST_PATH --load_vq
CUDA_VISIBLE_DEVICES=$(get_available_gpu) python metrics_large.py -m output/$CONFIG -t val_4
