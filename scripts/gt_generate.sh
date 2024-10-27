python tools/downsample.py -f data/GauU_Scene/LFLS/LFLS.ply -v 0.0035
python tools/downsample.py -f data/GauU_Scene/SMBU/smbu.ply -v 0.0035
python tools/downsample.py -f data/GauU_Scene/CUHK_UPPER_COLMAP/cuhksz_upper_campus.ply -v 0.0035
python tools/downsample.py -f data/matrix_city/point_cloud_ds20/aerial/Block_all.ply -v 0.0035
python tools/downsample.py -f data/matrix_city/point_cloud_ds20/street/Block_A.ply -v 0.0035 -s 100  # scale to fit the scene

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
