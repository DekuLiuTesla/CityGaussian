CONFIG="refine_ue_s2a_blockFcam_30k"
TEST_PATH="blockF_test20_1f2"

python refine.py --config config/$CONFIG.yaml
python render.py -m output/$CONFIG
python metrics_custom.py -m output/$CONFIG

# python render_fuse.py --config config/$CONFIG.yaml --custom_test --skip_train --skip_test
# python metrics_custom.py -m output/$CONFIG -t $TEST_PATH