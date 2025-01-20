# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

PROJECT=JointGS
TEST_PATH=data/matrix_city/aerial/test/block_all_test
declare -a run_args=(
    # "citygsv2_mc_aerial_sh2_trim_lod0"
    # "citygsv2_mc_aerial_sh2_trim_lod1"
    "citygsv2_mc_aerial_sh2_trim_lod2"
    "citygsv2_mc_aerial_sh2_trim_lod2_10geo"
)

for arg in "${run_args[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available."
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                                        --config configs/$arg.yaml \
                                        -n $arg \
                                        --logger wandb \
                                        --project $PROJECT \
                                        --data.train_max_num_images_to_cache 1024 &
            # Increment the port number for the next run
            ((port++))
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done
wait

for arg in "${run_args[@]}"; do
    while true; do
        gpu_id=$(get_available_gpu)
        if [[ -n $gpu_id ]]; then
            echo "GPU $gpu_id is available."
            CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
                                        --config outputs/$arg/config.yaml \
                                        --data.path $TEST_PATH \
                                        --data.parser.eval_image_select_mode ratio \
                                        --data.parser.eval_ratio 1.0 \
                                        --save_val
            
            CUDA_VISIBLE_DEVICES=$gpu_id python utils/gs2d_mesh_extraction.py \
                                        outputs/$arg \
                                        --voxel_size 0.01 \
                                        --sdf_trunc 0.04 \
                                        --depth_trunc 5.0
            
            CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run_mc.py \
                                                --scene Block_all_ds \
                                                --dataset-dir data/matrix_city/point_cloud_ds20/aerial \
                                                --ply-path "outputs/$arg/fuse_post.ply"

            # Increment the port number for the next run
            ((port++))
            # Allow some time for the process to initialize and potentially use GPU memory
            sleep 120
            break
        else
            echo "No GPU available at the moment. Retrying in 2 minute."
            sleep 120
        fi
    done
done
wait