CONFIG="org_rubble_all"
CUDA_VISIBLE_DEVICES=0 python train_large.py --config config/$CONFIG.yaml --port 7025&

CONFIG="org_residence_all"
CUDA_VISIBLE_DEVICES=1 python train_large.py --config config/$CONFIG.yaml --port 7026 &

CONFIG="org_building_all_lr2"
CUDA_VISIBLE_DEVICES=2 python train_large.py --config config/$CONFIG.yaml --port 7027 &

CONFIG="org_mc_aerial_block_all_lr2"
CUDA_VISIBLE_DEVICES=3 python train_large.py --config config/$CONFIG.yaml --port 7028 &

CONFIG="org_rubble_all"
TEST_PATH="data/mill19/rubble-pixsfm/val"
CUDA_VISIBLE_DEVICES=0 python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=0 python metrics_large.py -m output/$CONFIG -t val

CONFIG="org_residence_all"
TEST_PATH="data/urban_scene_3d/residence-pixsfm/val"
CUDA_VISIBLE_DEVICES=1 python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=1 python metrics_large.py -m output/$CONFIG -t val

CONFIG="org_building_all_lr2"
TEST_PATH="data/mill19/building-pixsfm/val"
CUDA_VISIBLE_DEVICES=2 python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=2 python metrics_large.py -m output/$CONFIG -t val

CONFIG="org_mc_aerial_block_all_lr2"
TEST_PATH="data/matrix_city/aerial/test/block_all_test"
CUDA_VISIBLE_DEVICES=3 python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=3 python metrics_large.py -m output/$CONFIG -t block_all_test