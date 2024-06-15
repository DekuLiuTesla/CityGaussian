# Function to get the id of an available GPU
get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

# # Downsample images
# python utils/image_downsample.py data/urban_scene_3d/sci-art-pixsfm/train/images --factor 4
# python utils/image_downsample.py data/urban_scene_3d/sci-art-pixsfm/val/images --factor 4

# # Generate appearance groups
# python utils/generate_image_apperance_groups.py data/urban_scene_3d/sci-art-pixsfm/train \
#     --image \
#     --name appearance_group_by_image
# python utils/generate_image_apperance_groups.py data/urban_scene_3d/sci-art-pixsfm/val \
#     --image \
#     --name appearance_group_by_image

NAME=campus-pixsfm-swag-grad

# Single GPU at the beginning
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
--config configs/large_scale_appearance_swag_campus.yaml \
--data.path data/urban_scene_3d/campus-pixsfm/train \
--data.params.colmap.down_sample_factor 4 \
--data.params.colmap.appearance_groups appearance_group_by_image \
-n $NAME \
--logger wandb \
--project JointGS \

gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.path data/urban_scene_3d/campus-pixsfm/val \
    --data.params.colmap.down_sample_factor 4 \
    --data.params.colmap.eval_image_select_mode ratio \
    --data.params.colmap.eval_ratio 1.0 \
    --model.save_val_output true \
    --model.correct_color true \


# # large scale dataset
NAME=sci-art-pixsfm-swag-grad8

# # Single GPU at the beginning
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
#     --config configs/large_scale_appearance_swag.yaml \
#     --data.path data/urban_scene_3d/sci-art-pixsfm/train \
#     --data.params.colmap.down_sample_factor 4 \
#     --data.params.colmap.appearance_groups appearance_group_by_image \
#     -n $NAME \
#     --logger wandb \
#     --project JointGS \

# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
#     --config outputs/$NAME/config.yaml \
#     --data.path data/urban_scene_3d/sci-art-pixsfm/val \
#     --data.params.colmap.down_sample_factor 4 \
#     --data.params.colmap.eval_image_select_mode ratio \
#     --data.params.colmap.eval_ratio 1.0 \
#     --model.save_val_output true \
#     --model.correct_color true \