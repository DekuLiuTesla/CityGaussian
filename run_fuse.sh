CONFIG="fuse_ue_aerial_20_block123Fcam_7k"
python fuse.py --config config/$CONFIG.yaml

# TEST_PATH="data/UE-collected/street/block2_5_cam"
python render_fuse.py --config config/$CONFIG.yaml
python metrics_custom.py -m output/$CONFIG

# TEST_PATH="data/UE-collected/aerial/blockF_test20_1f2"
# python render_fuse.py --config config/$CONFIG.yaml --custom_test $TEST_PATH  --skip_train --skip_test
# python metrics_custom.py -m output/$CONFIG -t blockF_test20_1f2