CONFIG="org_ssim_0.8_tandt_train"

python train_time.py --config config/$CONFIG.yaml
python render.py -m output/$CONFIG
python metrics.py -m output/$CONFIG