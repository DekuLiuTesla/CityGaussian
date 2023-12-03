CONFIG="org_ue_aerial_20_blockFcam"

python train_time.py --config config/$CONFIG.yaml
python render.py -m output/$CONFIG
python metrics.py -m output/$CONFIG