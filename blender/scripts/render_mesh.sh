# COARSE_NAME=citygs2d_sciart_coarse
# NAME=citygs2d_sciart
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python mesh.py --model_path outputs/$NAME --config_path outputs/$COARSE_NAME/config.yaml --voxel_size 0.4 --sdf_trunc 2 --depth_trunc 150


# python tools/generate_traj.py --config "outputs/citygs2d_building_coarse/config.yaml" \
#                               --mesh_path "outputs/citygs2d_building/mesh/merged_epoch=16-step=30000" \
#                               --scale_percentile 90

# cd blender
python render_sun.py --load_dir "../outputs/citygs2d_building/mesh/merged_epoch=16-step=30000" \
                     --mesh_file building_filtered.ply \
                     --config_dir render_cfgs/pixsfm \

# python render_sun.py --load_dir "../outputs/citygs2d_sciart/mesh/merged_epoch=11-step=30000" \
#                      --mesh_file sciart_d150_filtered.ply \
#                      --config_dir render_cfgs/pixsfm \

# python render_sun.py --load_dir "../outputs/citygs2d_residence/mesh/merged_epoch=12-step=30000" \
#                      --mesh_file residence_d150_filtered.ply \
#                      --config_dir render_cfgs/pixsfm \

# python render_sun.py --load_dir "../outputs/citygs2d_rubble/mesh/merged_epoch=19-step=30000" \
#                      --mesh_file rubble_filtered.ply \
#                      --config_dir render_cfgs/pixsfm \