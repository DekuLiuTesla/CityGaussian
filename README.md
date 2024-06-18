# Gaussian Splatting PyTorch Lightning Implementation
* <a href="#1-installation">Installation</a>
* <a href="#2-training">Training</a>
* <a href="#4-web-viewer">Web Viewer</a>
## Known issues
* Multi-GPU training can only be enabled after densification
## Features
* Multi-GPU/Node training (only after densification)
* Switch between diff-gaussian-rasterization and <a href="https://github.com/nerfstudio-project/gsplat">nerfstudio-project/gsplat</a>
* Multiple dataset types support
  * <a href="https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi">Blender (nerf_synthetic)</a>
  * Colmap
  * <a href="https://github.com/google/nerfies?tab=readme-ov-file#datasets">Nerfies</a>
  * <a href="https://github.com/facebookresearch/NSVF?tab=readme-ov-file#dataset">NSVF (Synthetic only)</a>
  * <a href="https://city-super.github.io/matrixcity/">MatrixCity</a>
  * <a href="https://www.cs.ubc.ca/~kmyi/imw2020/data.html">PhotoTourism</a>
* Dynamic object mask
* Appearance variation support
* Deformable Gaussians
  * <a href="https://ingra14m.github.io/Deformable-Gaussians/">Deformable 3D Gaussians</a>
  * <a href="https://guanjunwu.github.io/4dgs/index.html">4D Gaussian</a> (Viewer Only)
* <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting</a>
* <a href="https://lightgaussian.github.io/">LightGaussian</a>
* <a href="https://ty424.github.io/AbsGS.github.io/">AbsGS</a> / EfficientGS
* <a href="https://github.com/hbb1/2d-gaussian-splatting">2D Gaussian Splatting</a>
* Load arbitrary number of images without OOM
* Interactive web viewer
  * Load multiple models
  * Model transform
  * Scene editor
  * Video camera path editor
* Video renderer
## 1. Installation
### 1.1. Clone repository

```bash
# clone repository
git clone --recursive https://github.com/yzslab/gaussian-splatting-lightning.git
cd gaussian-splatting-lightning
```

* If you forgot the `--recursive` options, you can run below git commands after cloning:

  ```bash
   git submodule sync --recursive
   git submodule update --init --recursive --force
  ```

### 1.2. Create virtual environment

```bash
# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl
```

### 1.3. Install PyTorch
* Tested on `PyTorch==2.0.1`
* You must install the one match to the version of your nvcc (nvcc --version)
* For CUDA 11.8

  ```bash
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

### 1.4. Install requirements

```bash
pip install -r requirements.txt
```

### 1.5. Install optional packages
* If you want to train with appearance variation images

  ```bash
  pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
  ```

* If you want to use nerfstudio-project/gsplat
  * Vanilla version

    ```bash
    pip install gsplat==0.1.11
    ```

  * If you need MipSplatting, LightGaussian

    ```bash
    pip install git+https://github.com/yzslab/gsplat.git
    ```

## 2. Training
### 2.1. Basic command
```bash
python main.py fit \
    --data.path DATASET_PATH \
    -n EXPERIMENT_NAME
```
It can detect some dataset type automatically. You can also specify type with option `--data.type`. Possible values are: `colmap`, `blender`, `nsvf`, `nerfies`, `matrixcity`, `phototourism`.

<b>[NOTE]</b> By default, only checkpoint files will be produced on training end. If you need ply file in vanilla 3DGS's format (can be loaded by SIBR_viewer or some WebGL/GPU based viewer):
  * [Option 1]: Convert checkpoint file to ply: `python utils/ckpt2ply.py TRAINING_OUTPUT_PATH`, e.g.:
    * `python utils/ckpt2ply.py outputs/lego`
    * `python utils/ckpt2ply.py outputs/lego/checkpoints/epoch=300-step=30000.ckpt`
  * [Option 2]: Start training with option: `--model.save_ply true`
### 2.2. Some useful options
* Run training with web viewer
```bash
python main.py fit \
    --viewer \
    ...
```
* It is recommended to use config file `configs/blender.yaml` when training on blender dataset.
```bash
python main.py fit \
    --config configs/blender.yaml \
    ...
```
* With mask (colmap dataset only)
  * You may need to undistort mask images too: <a href="https://github.com/yzslab/gaussian-splatting-lightning/blob/main/utils/colmap_undistort_mask.py">utils/colmap_undistort_mask.py</a>
```bash
# the requirements of mask
#   * must be single channel
#   * zero(black) represent the masked pixel (won't be used to supervise learning)
#   * the filename of the mask file must be image filename + '.png', 
#     e.g.: the mask of '001.jpg' is '001.jpg.png'
--data.params.colmap.mask_dir MASK_DIR_PATH
```
* Use downsampled images (colmap dataset only)

You can use `utils/image_downsample.py` to downsample your images, e.g. 4x downsample: `python utils/image_downsample.py PATH_TO_DIRECTORY_THAT_STORE_IMAGES --factor 4`
```bash
# it will load images from `images_4` directory
--data.params.colmap.down_sample_factor 4
```
* Load large dataset without OOM
```bash
--data.params.train_max_num_images_to_cache 1024
```
* Enable appearance model to train on appearance variation images (colmap dataset only)
```bash
# 1. Generate appearance groups
python generate_image_apperance_groups.py PATH_TO_DATASET \
    --camera \
    --name appearance_group_by_camera
    
# 2. Enable appearance model
python main.py fit \
    ... \
    --model.renderer AppearanceMLPRenderer \
    --data.params.colmap.appearance_groups appearance_group_by_camera \
    ...
```

### 2.3. Use <a href="https://github.com/nerfstudio-project/gsplat">nerfstudio-project/gsplat</a>
Make sure that command `which nvcc` can produce output, or gsplat will be disabled automatically.
```bash
python main.py fit \
    --config configs/gsplat.yaml \
    ...
```

### 2.4. Multi-GPU training
<b>[NOTE]</b> Multi-GPU training can only be enabled after densification. You can start a single GPU training at the beginning, and save a checkpoint after densification finishing. Then resume from this checkpoint and enable multi-GPU training.

You will get improved PSNR and SSIM with more GPUs:
![image](https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/06e91e71-5068-46ce-b169-524a069609bf)


```bash
# Single GPU at the beginning
python main.py fit \
    --config ... \
    --data.path DATASET_PATH \
    --model.gaussian.optimization.densify_until_iter 15000 \
    --max_steps 15000
# Then resume, and enable multi-GPU
python main.py fit \
    --config ... \
    --trainer configs/ddp.yaml \
    --data.path DATASET_PATH \
    --max_steps 30000 \
    --ckpt_path last  # find latest checkpoint automatically, or provide a path to checkpoint file
```

### 2.5. <a href="https://ingra14m.github.io/Deformable-Gaussians/">Deformable 3D Gaussians</a>
<video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/177b3fbf-fdd2-490f-b446-433a4d929502"></video>

```bash
python main.py fit \
    --config configs/deformable_blender.yaml \
    --data.path ...
```

### 2.6. <a href="https://niujinshuchong.github.io/mip-splatting/">Mip-Splatting</a>
```bash
python main.py fit \
    --config configs/mip_splatting_gsplat.yaml \
    --data.path ...
```

### 2.7. <a href="https://lightgaussian.github.io/">LightGaussian</a>
* Prune & finetune only currently
* Train & densify & prune

  ```bash
  ... fit \
      --config configs/light_gaussian/train_densify_prune-gsplat.yaml \
      --data.path ...
  ```

* Prune & finetune (make sure to use the same hparams as the input model used)

  ```bash
  ... fit \
      --config configs/light_gaussian/prune_finetune-gsplat.yaml \
      --data.path ... \
      ... \
      --ckpt_path YOUR_CHECKPOINT_PATH
  ```
  
### 2.8. <a href="https://ty424.github.io/AbsGS.github.io/">AbsGS</a> / EfficientGS
```bash
... fit \
    --config configs/gsplat-absgrad.yaml \
    --data.path ...
```

### 2.9. <a href="https://surfsplatting.github.io/">2D Gaussian Splatting</a>
* Install `diff-surfel-rasterization` first
  ```bash
  pip install git+https://github.com/hbb1/diff-surfel-rasterization.git@3a9357f6a4b80ba319560be7965ed6a88ec951c6
  ```

* Then start training
  ```bash
  ... fit \
      --config configs/vanilla_2dgs.yaml \
      --data.path ...
  ```

## 3. Evaluation

### Evaluate on validation set
```bash
python main.py validate \
    --config outputs/lego/config.yaml
```

### On test set
```bash
python main.py test \
    --config outputs/lego/config.yaml
```

### On train set
```bash
python main.py validate \
    --config outputs/lego/config.yaml \
    --val_train
```

### Save images that rendered during evaluation/test
```bash
python main.py <validate or test> \
    --config outputs/lego/config.yaml \
    --save_val
```
Then you can find the images in `outputs/lego/<val or test>`.

## 4. Web Viewer
| Transform | Camera Path | Edit |
| --- | --- | --- |
| <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/de1ff3c3-a27a-4600-8c76-ab6551df6fca"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/3f87243d-d9a1-41e2-9d51-225735925db4"></video> | <video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/7cf0ccf2-44e9-4fc9-87cc-740b7bbda488"></video> |


### 4.1 Basic usage
* Also works for <a href="https://github.com/graphdeco-inria/gaussian-splatting">graphdeco-inria/gaussian-splatting</a>'s ply output
```bash
python viewer.py TRAINING_OUTPUT_PATH
# e.g.: 
#   python viewer.py outputs/lego/
#   python viewer.py outputs/lego/checkpoints/epoch=300-step=30000.ckpt
#   python viewer.py outputs/lego/baseline/point_cloud/iteration_30000/point_cloud.ply  # only works with VanillaRenderer
```
### 4.2 Load multiple models and enable transform options
```bash
python viewer.py \
    outputs/garden \
    outputs/lego \
    outputs/Synthetic_NSVF/Palace/point_cloud/iteration_30000/point_cloud.ply \
    --enable_transform
```

### 4.3 Load model trained by other implementations
<b>[NOTE]</b> The commands in this section only design for third-party outputs

* <a href="https://github.com/ingra14m/Deformable-3D-Gaussians">ingra14m/Deformable-3D-Gaussians</a>

```bash
python viewer.py \
    Deformable-3D-Gaussians/outputs/lego \
    --vanilla_deformable \
    --reorient disable  # change to enable when loading real world scene
```

* <a href="https://github.com/hustvl/4DGaussians">hustvl/4DGaussians</a>

```bash
python viewer.py \
    4DGaussians/outputs/lego \
    --vanilla_gs4d
```

* <a href="https://github.com/hbb1/2d-gaussian-splatting">hbb1/2d-gaussian-splatting</a>
```bash
# Install `diff-surfel-rasterization` first
pip install git+https://github.com/hbb1/diff-surfel-rasterization.git@3a9357f6a4b80ba319560be7965ed6a88ec951c6
# Then start viewer
python viewer.py \
    2d-gaussian-splatting/outputs/Truck \
    --vanilla_gs2d
```

## 5. F.A.Q.
<b>Q: </b> The viewer shows my scene in unexpected orientation, how to rotate the camera, like the `U` and `O` key in the SIBR_viewer?

<b>A: </b> Check the `Orientation Control` on the right panel, rotate the camera frustum in the scene to the orientation you want, then click `Apply Up Direction`.
<video src="https://github.com/yzslab/gaussian-splatting-lightning/assets/564361/7e9198b5-d853-4800-aac2-1774640a8874"></video>

<br/>

Besides: You can also click the 'Reset up direction' button. Then the viewer will use your current orientation as the reference.
 * First use mouse to rotate your camera to the orientation you want
 * Then click the 'Reset up direction' button


##

<b>Q: </b> The web viewer is slow (or low fps, far from real-time).

<b>A: </b> This is expected because of the overhead of the image transfer over network. You can get around 10fps in 1080P resolution, which is enough for you to view the reconstruction quality.

# License
This repository is licensed under MIT license. Except some thirdparty dependencies (e.g. files in `submodules` directory), files and codes copied from other repositories, which are separately licensed.
```text
MIT License

Copyright (c) 2023 yzslab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
