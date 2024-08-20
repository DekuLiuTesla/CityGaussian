## A. Arrange Data of the Scene
Firstly, please arrange the data of your scene in 3DGS-style structure. Suppose that you have a scene named `my_scene`, then its images and COLMAP results should be set like:

```
├── data
│   ├── my_scene
│   │   ├── train
│   │   │   ├── images
│   │   │   ├── sparse
│   │   │   │   ├── 0
│   │   │   │   │   ├── cameras.bin
│   │   │   │   │   ├── points3D.bin
│   │   │   │   │   ├── images.bin
│   │   ├── test
│   │   │   ├── images
│   │   │   ├── sparse
│   │   │   │   ├── 0
│   │   │   │   │   ├── cameras.bin
│   │   │   │   │   ├── images.bin
│   │   │   │   │   ├── points3D.bin
```

## B. Prepare Configurations
1. Prepare config for pretrain as suggested in `config/my_scene_coarse.yaml`. You can consider this step as a normal 3DGS training. Adjust `source_path` and `resolution` according to your need.

2. Prepare config for finetuning as suggested in `config/my_scene.yaml`. This step enables block-wise tuning. Adjust `source_path` and `resolution` as you do in previous step. For `aabb` and `ssim_threshold`, you can directly apply our default setting. Default `aabb` selects central 1/3 region of interested area. For `block_dim`, you can set it according to length-width ratio of interested area. For instance, if you want to reconstruct a building and it's around 800m x 600m, then you can use a block dimension of 4 x 3 x 1 (Suppose z is height dimension). **Ensure the performance has been improved upon coarse model before moving to next step**.

3. Prepare configs for LoD generation according to `LargeLightGaussian/scripts/run_prune_finetune_my_scene.sh`, `LargeLightGaussian/scripts/run_distill_finetune_my_scene.sh`, and `LargeLightGaussian/scripts/run_vectree_quantize_my_scene.sh`. For more details, please refer to [LightGaussian paper](https://arxiv.org/abs/2311.17245).
    - For pruning, you mainly need to consider `data_path`, `run_args`, `prune_percents` and `prune_names`. The higher `prune_percents`, the lower model size.
    - For distill, you mainly need to consider `data_path` and `run_args`, as you do in previous step.
    - For quantization, you mainly need to consider `SCENES`. It's the same as `run_args` of previous step.
    - Besides, if you use large-scale scene under real size, i.e. take meter as distance unit, please set learning rate similar to configs of `Rubble` dataset.

4. Prepare config for LoD rendering as suggested in `config/my_scene_lod.yaml`. The main difference between `my_scene.yaml` and `my_scene_lod.yaml` are model, `sh_degree`, and `lod_configs`. Note that after LightGaussian compression, the SH degree has decreased to 2. Adjust `lod_configs` to the output name of your compressed models.

## C. Adjust the Scripts
1. For `scripts/run_citygs.sh`, adjust `TEST_PATH` to `data/my_scene/test`, `COARSE_CONFIG` to `my_scene_coarse`, `CONFIG` to `my_scene`, `out_name` ot `test`. Suppose you adopt block dimension of M x N x S in step 2 of section B, then set `max_block_id` to `M*N*S-1`.

2. For `scripts/run_citygs_lod.sh`, adjust `TEST_PATH` and `out_name` as you do in previous step. Then change `CONFIG` to `my_scene_lod`.

3. Run scripts in order:
    ```
    bash scripts/run_citygs.sh
    cd LargeLightGaussian
    bash scripts/run_prune_finetune_my_scene.sh
    bash scripts/run_distill_finetune_my_scene.sh
    bash scripts/run_vectree_quantize_my_scene.sh
    cd ..
    bash scripts/run_citygs_lod.sh
    ```