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
W