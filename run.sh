CONFIG="org_mc_aerial_block9"

python train_large.py --config config/$CONFIG.yaml

TEST_PATH="data/matrix_city/aerial/test/block_9_test"
python render_fuse.py --config config/$CONFIG.yaml --custom_test $TEST_PATH  --skip_train --skip_test
python metrics_custom.py -m output/$CONFIG -t block_9_test
# python render_large.py -m output/$CONFIG
# python metrics.py -m output/$CONFIG