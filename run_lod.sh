CONFIG="lod_mc_aerial_block3_vox_ft_all"

# train coarse global GS
CUDA_VISIBLE_DEVICES=1 python train_large.py --config config/$CONFIG.yaml --save_iterations 7000 15000

# train GS cell
# for num in {0..15}  
# do  
# python train_large.py --config config/$CONFIG.yaml --block_id $num
# done 

TEST_PATH="data/matrix_city/aerial/test/block_3_test"
python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
python metrics_custom.py -m output/$CONFIG -t block_3_test