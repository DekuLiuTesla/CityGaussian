CONFIG="fuse_ue_aerial_20_blockFcam_7k"

# python convert.py -s data/UE-collected/aerial/block1_6
# python convert_cam.py -s data/UE-collected/aerial/blockF_20_cam

# python fuse.py --config config/$CONFIG.yaml
python render.py -m output/$CONFIG
python metrics.py -m output/$CONFIG