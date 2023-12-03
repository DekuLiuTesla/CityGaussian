CONFIG="fuse_ue_aerial_20_blockFcam_7k_reset2k"

python fuse.py --config config/$CONFIG.yaml
python render_fuse.py --config config/$CONFIG.yaml --source_test
python metrics_custom.py -m output/$CONFIG -t test_src

# python fuse.py --config config/$CONFIG.yaml
# python render_fuse.py --config config/$CONFIG.yaml
# python metrics_custom.py -m output/$CONFIG
