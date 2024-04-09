# CONFIG="org_mc_aerial_block3_distorted"
# CUDA_VISIBLE_DEVICES=4 python train_large.py --config config/$CONFIG.yaml

# train GS cell
CONFIG="block_mc_aerial_block_all_lr_c36_loss_8_num_10w_lr2"
CUDA_VISIBLE_DEVICES=0 python data_partition.py --config config/$CONFIG.yaml
# select num range according to block number
for num in {0..3}  
do
let _num=num*8+0
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1024 &
let _num=num*8+1
CUDA_VISIBLE_DEVICES=1 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1025 &
let _num=num*8+2
CUDA_VISIBLE_DEVICES=2 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1026 &
let _num=num*8+3
CUDA_VISIBLE_DEVICES=3 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1027 &
let _num=num*8+4
CUDA_VISIBLE_DEVICES=4 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1028 &
let _num=num*8+5
CUDA_VISIBLE_DEVICES=5 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1029 &
let _num=num*8+6
CUDA_VISIBLE_DEVICES=6 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1030 &
let _num=num*8+7
CUDA_VISIBLE_DEVICES=7 WANDB_MODE=offline python train_large.py --config config/$CONFIG.yaml --block_id $_num --port 1031 &
wait
done

python merge.py --config config/$CONFIG.yaml

TEST_PATH="data/matrix_city/aerial/test/block_all_test"
CUDA_VISIBLE_DEVICES=3 python render_large.py --config config/$CONFIG.yaml --custom_test $TEST_PATH
CUDA_VISIBLE_DEVICES=3 python metrics_custom.py -m output/$CONFIG -t block_all_test