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