CONFIG="fuse_ue_aerial_20_blockFcam_7k"

python fuse.py --config config/$CONFIG.yaml
python render_fuse.py --config config/$CONFIG.yaml --custom_test --skip_train --skip_test
python metrics_custom.py -m output/$CONFIG -t blockF_test20_1f2

# python fuse.py --config config/$CONFIG.yaml
# python render_fuse.py --config config/$CONFIG.yaml
# python metrics_custom.py -m output/$CONFIG