get_available_gpu() {
  local mem_threshold=500
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -v threshold="$mem_threshold" -F', ' '
  $2 < threshold { print $1; exit }
  '
}

NAME=mc_street_part_30K
VIEW=street

# train
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
    --config configs/mc_street_part_30K.yaml \
    --data.path data/matrix_city/aerial_street_fusion/$VIEW/train \
    -n $NAME \
    --logger wandb \
    --project JointGS \

# mgu train
# gpu_id=$(get_available_gpu)
# echo "GPU $gpu_id is available."
# CUDA_VISIBLE_DEVICES=$gpu_id python main.py fit \
#     --config configs/mc_fuse_w_bkgd.yaml \
#     --data.path data/matrix_city/aerial_street_fusion/$VIEW/train \
#     -n $NAME \
#     --logger wandb \
#     --project JointGS \
#     --max_steps 30000 \
  
# Then resume, and enable multi-GPU
# CUDA_VISIBLE_DEVICES="0,1" python main.py fit \
#     --config configs/mc_fuse_w_bkgd.yaml \
#     --trainer configs/ddp.yaml \
#     --data.path data/matrix_city/aerial_street_fusion/$VIEW/train \
#     -n $NAME \
#     --logger wandb \
#     --project JointGS \
#     --max_steps 60000 \
#     --ckpt_path last  # find latest checkpoint automatically, or provide a path to checkpoint file

# test
gpu_id=$(get_available_gpu)
echo "GPU $gpu_id is available."
CUDA_VISIBLE_DEVICES=$gpu_id python main.py test \
    --config outputs/$NAME/config.yaml \
    --data.path data/matrix_city/aerial_street_fusion/$VIEW/test \
    --data.params.colmap.eval_image_select_mode ratio \
    --data.params.colmap.eval_ratio 1.0 \
    --model.save_val_output true \