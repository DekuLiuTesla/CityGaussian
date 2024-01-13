# train coarse global GS
# CONFIG="org_mc_aerial_block9"
# CUDA_VISIBLE_DEVICES=1 python train_large.py --config config/$CONFIG.yaml --port 314

# train GS cell
CONFIG="block_mc_aerial_block9_lod"
for num in {0..3}  
do  
let _num=num*4+0
CUDA_VISIBLE_DEVICES=0 python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 0 &
let _num=num*4+1
CUDA_VISIBLE_DEVICES=1 python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1 &
let _num=num*4+2
CUDA_VISIBLE_DEVICES=2 python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 2 &
let _num=num*4+3
CUDA_VISIBLE_DEVICES=3 python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 3 &
wait
done 

# num=9
# CUDA_VISIBLE_DEVICES=3 python train_large.py --config config/$CONFIG.yaml --block_id $num --port 2012

TEST_PATH="data/matrix_city/aerial/test/block_9_test"
CUDA_VISIBLE_DEVICES=3 python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=3 python metrics_custom.py -m output/$CONFIG -t block_9_test