CONFIG="org_m360_bicycle_grad"

python train_time.py --config config/$CONFIG.yaml
python render.py -m output/$CONFIG
python metrics.py -m output/$CONFIG