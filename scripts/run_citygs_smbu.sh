# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

COARSE_NAME=citygsv2_smbu_coarse_sh2
NAME=citygsv2_smbu_sh2_trim
PROJECT=YOUR_PROJECT_NAME  # Change to your project name

# ============================================= downsample images =============================================
# python utils/image_downsample.py data/GauU_Scene/SMBU/images --factor 3.4175

# ===================================== generate depth with depth-anything-V2 =================================
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python utils/estimate_dataset_depths.py \
#                                     data/GauU_Scene/SMBU \
#                                     -d 3.4175 \

# ============================================= train&eval coarse model =============================================
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
                                    --config configs/$COARSE_NAME.yaml \
                                    -n $COARSE_NAME \
                                    --logger wandb \
                                    --project $PROJECT \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$COARSE_NAME/config.yaml \
    --save_val \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/gs2d_mesh_extraction.py \
                                    outputs/$COARSE_NAME \
                                    --voxel_size 0.01 \
                                    --sdf_trunc 0.04 \
                                    --depth_trunc 2.0

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run.py \
                                    --scene SMBU_ds \
                                    --dataset-dir data/geometry_gt/SMBU \
                                    --transform-path data/geometry_gt/SMBU/transform.txt \
                                    --ply-path "outputs/$COARSE_NAME/fuse_post.ply"

# ============================================= generate partition =============================================
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/partition_citygs.py --config_path configs/$NAME.yaml --force  # --reorient

# =========================================== train&eval tuned model ===========================================
python utils/train_citygs_partitions.py -n $NAME -p $PROJECT

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/merge_citygs_ckpts.py outputs/$NAME \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
                                    --config configs/$NAME.yaml \
                                    -n $NAME \
                                    --data.parser.split_mode experiment \
                                    --data.parser.eval_image_select_mode ratio \
                                    --data.parser.eval_ratio 0.1 \
                                    --save_val \
                                    --test_speed \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python utils/gs2d_mesh_extraction.py \
                                    outputs/$NAME \
                                    --voxel_size 0.01 \
                                    --sdf_trunc 0.04 \
                                    --depth_trunc 2.0

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python tools/eval_tnt/run.py \
                                    --scene SMBU_ds \
                                    --dataset-dir data/geometry_gt/SMBU \
                                    --transform-path data/geometry_gt/SMBU/transform.txt \
                                    --ply-path "outputs/$NAME/fuse_post.ply"

# python tools/block_wandb_sync.py --output_path outputs/$NAME  # Synchronize results to wandb if needed

# ================================= remove block results (if you find result OK) ================================
# rm -rf outputs/$NAME/blocks/block_*/checkpoints

# ============================================= vector quantization =============================================
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python tools/vectree_lightning.py \
#                                     --model_path outputs/$NAME \
#                                     --save_path outputs/$NAME/vectree \
#                                     --sh_degree 2 \
                                    # --skip_quantize \
                                    # --no_save_ply \
