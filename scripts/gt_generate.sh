python utils/downsample_pcd.py -f data/GauU_Scene/LFLS/LFLS.ply -v 0.0035
python utils/downsample_pcd.py -f data/GauU_Scene/SMBU/smbu.ply -v 0.0035
python utils/downsample_pcd.py -f data/GauU_Scene/CUHK_UPPER_COLMAP/cuhksz_upper_campus.ply -v 0.0035
python utils/downsample_pcd.py -f data/matrix_city/point_cloud_ds20/aerial/Block_all.ply -v 0.0035
python utils/downsample_pcd.py -f data/matrix_city/point_cloud_ds20/street/Block_A.ply -v 0.0035 -s 100  # scale to fit the scene

python utils/generate_crop_volume.py --ply_path data/GauU_Scene/LFLS/LFLS_ds.ply \
                                     --transform_path data/GauU_Scene/Downsampled/LFLS/transform.txt \
                                     --data_path data/GauU_Scene/LFLS \
                                     --split_mode experiment \
                                     --down_sample_factor 3.4175 \
                                     --vis_threshold 95 \

python utils/generate_crop_volume.py --ply_path data/GauU_Scene/SMBU/SMBU_ds.ply \
                                     --transform_path data/GauU_Scene/Downsampled/SMBU/transform.txt \
                                     --data_path data/GauU_Scene/SMBU \
                                     --split_mode experiment \
                                     --down_sample_factor 3.4175 \
                                     --vis_threshold 125 \

python utils/generate_crop_volume.py --ply_path data/GauU_Scene/CUHK_UPPER_COLMAP/CUHK_UPPER_COLMAP_ds.ply \
                                     --transform_path data/GauU_Scene/Downsampled/CUHK_UPPER_COLMAP/transform.txt \
                                     --data_path data/GauU_Scene/CUHK_UPPER_COLMAP \
                                     --split_mode experiment \
                                     --down_sample_factor 3.4175 \
                                     --vis_threshold 90 \

python utils/generate_crop_volume.py --ply_path data/matrix_city/point_cloud_ds20/aerial/Block_all_ds.ply \
                                     --data_path data/matrix_city/aerial/train/block_all \
                                     --down_sample_factor 1.2 \
                                     --vis_threshold 125 \

python utils/generate_crop_volume.py --ply_path data/matrix_city/point_cloud_ds20/street/Block_A_ds.ply \
                                     --data_path data/matrix_city/street/train/block_A \
                                     --vis_threshold 815 \  # several boundary points needs to be adjusted

python tools/generate_traj.py --config outputs/citygs2d_lfls_coarse_lnorm4_wo_vast_sep_depth_init_5/config.yaml
python tools/generate_traj.py --config outputs/citygs2d_smbu_coarse_lnorm4_wo_vast_sep_depth_init_5/config.yaml
python tools/generate_traj.py --config outputs/citygs2d_upper_coarse_lnorm4_wo_vast_sep_depth_init_5/config.yaml
python tools/generate_traj.py --config outputs/citygs2d_mc_aerial_coarse_lnorm4_wo_vast_sep_depth_init_5/config.yaml --data_path data/matrix_city/aerial/test/block_all_test --train
python tools/generate_traj.py --config outputs/citygs2d_mc_street_coarse_lnorm4_wo_vast_sep_depth_init_5_ubd1e1/config.yaml --data_path data/matrix_city/street/test/block_A_test --train

python tools/render_traj.py --config outputs/citygs2d_lfls_coarse_lnorm4_wo_vast_sep_depth_init_5/config.yaml \
                            --ckpt_path "outputs/citygs2d_lfls_lnorm4_wo_vast_sep_depth_trim/checkpoints/epoch=32-step=30000.ckpt" \
                            --data_path data/GauU_Scene/LFLS \
                            --scale_percentile 70 --n_frames 480 \
                            --pitch 40 --y_shift 1 --filter --train

python tools/render_traj.py --config outputs/citygs2d_smbu_coarse_lnorm4_wo_vast_sep_ssim_depth_init_5_v6/config.yaml \
                            --ckpt_path "outputs/citygs2d_smbu_lnorm4_wo_vast_sep_ssim_depth_trim_v6/checkpoints/epoch=60-step=30000.ckpt" \
                            --data_path data/GauU_Scene/SMBU \
                            --scale_percentile 85 --n_frames 480 --vox_grid 50 \
                            --pitch 40 --y_shift 1 --filter --train 

python tools/render_traj.py --config outputs/citygs2d_upper_coarse_lnorm4_wo_vast_sep_ssim_depth_init_5/config.yaml \
                            --ckpt_path "outputs/citygs2d_upper_lnorm4_wo_vast_sep_ssim_depth_trim/checkpoints/epoch=48-step=30000.ckpt" \
                            --data_path data/GauU_Scene/CUHK_UPPER_COLMAP \
                            --scale_percentile 78 --n_frames 480 \
                            --pitch 40 --filter --train 

python tools/render_traj.py --config outputs/citygs2d_mc_aerial_coarse_lnorm4_wo_vast_sep_depth_init_5/config.yaml \
                            --ckpt_path "outputs/citygs2d_mc_aerial_lnorm4_wo_vast_sep_depth/checkpoints/epoch=6-step=30000.ckpt" \
                            --data_path data/matrix_city/aerial/train/block_all \
                            --scale_percentile 97 --n_frames 960 \
                            --pitch 45 --filter --std_ratio 5.0 --train 

python tools/render_traj.py --config outputs/citygs2d_rubble_coarse_lnorm4_wo_vast_sep_ssim_depth_init_5_v6/config.yaml \
                            --ckpt_path "outputs/citygs2d_rubble_lnorm4_wo_vast_sep_ssim_depth_trim_v6/checkpoints/epoch=19-step=30000.ckpt" \
                            --data_path data/mill19/rubble-pixsfm/trai \
                            --scale_percentile 80 --n_frames 480 \
                            --pitch 40 --x_shift 25 --y_shift 10 --filter --train