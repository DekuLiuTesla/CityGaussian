CONFIG="fuse_ue_aerial_20_blockFcam_7k_reset2k"

# python convert.py -s data/UE-collected/aerial/block1_6
# python convert_cam.py -s data/UE-collected/aerial/blockF_test20_1f2

# python fuse.py --config config/$CONFIG.yaml
python render_fuse.py --config config/$CONFIG.yaml --source_test
python metrics_custom.py -m output/$CONFIG -t test_src

# python fuse.py --config config/$CONFIG.yaml
# python render_fuse.py --config config/$CONFIG.yaml
# python metrics_custom.py -m output/$CONFIG