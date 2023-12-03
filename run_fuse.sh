CONFIG="cat_ue_aerial_20_blockFcam"

# python train_time.py --config config/$CONFIG.yaml
python render_fuse.py --config config/$CONFIG.yaml
python metrics.py -m output/$CONFIG