get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# train coarse global gaussian model
CONFIG="org_mc_aerial_block_all_lr2"
CUDA_VISIBLE_DEVICES=$(get_available_gpu) python train_large.py --config config/$CONFIG.yaml

# train CityGaussian
CONFIG="block_mc_aerial_block_all_lr_c36_loss_8"
# obtain data partitioning
CUDA_VISIBLE_DEVICES=$(get_available_gpu) python data_partition.py --config config/$CONFIG.yaml

# optimize each block
port=4041
for num in {0..8}; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available. Starting training block '$num'"
            CUDA_VISIBLE_DEVICES=$gpu_id WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $num --port $port &
            # Increment the port number for the next run
            ((port++))
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 180
            break
        else
            echo "No GPU available at the moment. Retrying in 3 minute."
            sleep 180
        fi
    done
done
wait

# merge the blocks
CUDA_VISIBLE_DEVICES=$(get_available_gpu) python merge.py --config config/$CONFIG.yaml

# rendering and evaluation
TEST_PATH="data/matrix_city/aerial/test/block_all_test"
CUDA_VISIBLE_DEVICES=$(get_available_gpu) python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=$(get_available_gpu) python metrics_large.py -m output/$CONFIG -t block_all_test