<br>
<p align="center">
<h1 align="center"><strong>[ECCV2024] CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians</strong></h1>
  <p align="center">
    Yang Liu&emsp;
    He Guan&emsp;
    Chuanchen Luo&emsp;
    Lue Fan&emsp;
    Naiyan Wang&emsp;
    Junran Peng&emsp;
    Zhaoxiang Zhang&emsp;
    <br>
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
    <br>
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/Paper-%F0%9F%93%96-blue)](https://arxiv.org/pdf/2404.01133)
[![](https://img.shields.io/badge/Project-%F0%9F%9A%80-blue)](https://dekuliutesla.github.io/citygs/)
![GitHub Repo stars](https://img.shields.io/github/stars/DekuLiuTesla/CityGaussian)

</div>

The advancement of real-time 3D scene reconstruction and novel view synthesis has been significantly propelled by 3D Gaussian Splatting (3DGS). However, effectively training large-scale 3DGS and rendering it in real-time across various scales remains challenging. This paper introduces CityGaussian (CityGS), which employs a novel divide-and-conquer training approach and Level-of-Detail (LoD) strategy for efficient large-scale 3DGS training and rendering. Specifically, the global scene prior and adaptive training data selection enables efficient training and seamless fusion. Based on fused Gaussian primitives, we generate different detail levels through compression, and realize fast rendering across various scales through the proposed block-wise detail levels selection and aggregation strategy. Extensive experimental results on large-scale scenes demonstrate that our approach attains state-of-the-art rendering quality, enabling consistent real-time rendering of large-scale scenes across vastly different scales. **Welcome to visit our [Project Page](https://dekuliutesla.github.io/citygs/)**.

<!-- ![Teaser](assets/teaser.jpg) -->

<div style="text-align: center;">
    <img src="assets/Teaser.png" alt="Dialogue_Teaser" width=100% >
</div>

## 📰 News
**[2024.08.20]** Updates [Custom Dataset Instructions](doc/custom_dataset.md)! 

**[2024.08.05]** Our code is now available! Welcome to try it out!

**[2024.07.18]** Camera Ready version now can be accessed through arXiv. More insights are included.

## 🥏 Model of CityGaussian

This repository contains the official implementation of the paper ["CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians"](https://arxiv.org/pdf/2404.01133). Star ⭐ us if you like it!
### Training Pipeline
<p align="center">
  <img src="assets/Train.png" align="center" width="100%">
</p>

### Rendering Pipeline
<p align="center">
  <img src="assets/Render.png" align="center" width="100%">
</p>

## 🔧 Usage

Note that the configs for five large-scale scenes: MatrixCity, Rubble, Building, Residence and Sci-Art has been prepared in `config` folder. Data of these datasets can be prepared according to [Data Preparation](doc/data_preparation.md). For COLMAP, we recommend to directly use our generated results:

- **Google Drive**: https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1zX34zftxj07dCM1x5bzmbA?pwd=1t6r

### Installation
#### a. Clone the repository
```bash
# clone repository
git clone --recursive https://github.com/DekuLiuTesla/CityGaussian.git
cd CityGaussian
mkdir data  # store your dataset here
mkdir output  # store your output here
```

#### b. Create virtual environment
```bash
# create virtual environment
conda create -yn citygs python=3.9 pip
conda activate citygs
```

#### c. Install PyTorch
* Tested on `PyTorch==2.0.1`
* You must install the one match to the version of your nvcc (nvcc --version)
* For CUDA 11.8

  ```bash
  pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
  ```

#### d. Install requirements
```bash
pip install -r requirements.txt
```

#### e. Install tailored LightGaussian for LoD
```bash
cd LargeLightGaussian
pip install submodules/compress-diff-gaussian-rasterization
ln -s /path/to/data /path/to/LargeLightGaussian/data
ln -s /path/to/output /path/to/LargeLightGaussian/output
cd ..
```
### Prepare Config Files
If you use your own dataset, please follow instruction in [Custom Dataset Instructions](doc/custom_dataset.md) to prepare. We also prepared templates in `./config` and `LargeLightGaussian/scripts`.

### Training and Vanilla Rendering
To train a scene, config the hyperparameters of pretraining and finetuning stage with your yaml file, then replace the `COARSE_CONFIG` and `CONFIG` in `run_citygs.sh`. The `max_block_id`, `out_name`, and `TEST_PATH` in `run_citygs.sh` should be set according to your dataset as well. Then you can train your scene by simply using:
```bash
bash scripts/run_citygs.sh
```
This script will also render and evaluate the result without LoD.

### Rendering with LoD
First, the LoD generation is realized by the following command:
```bash
cd LargeLightGaussian
bash scripts/run_prune_finetune_$your_scene.sh
bash scripts/run_distill_finetune_$your_scene.sh
bash scripts/run_vectree_quantize_$your_scene.sh
cd ..
```
After that, configure the LoD setting in another yaml file. Then replace `CONFIG`, `TEST_PATH`, and `out_name` with yours in `run_citygs_lod.sh`. Then you can render the scene with LoD by using:
```bash
bash scripts/run_citygs_lod.sh
```
Note that the LoD selection is now based on Nyquist sampling rate instead of manually defined distance threshold. This modification enables better generalization and anti-aliasing performance.

### Viewer
We borrowed Web viewer from [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning). Take the scene Rubble as an example. To render the scene with no LoD, you can use the following command:
```bash
python viewer.py output/rubble_c9_r4
```
To render the scene with LoD, you can use the following command:
```bash
# copy cameras.json first for direction initialization
cp output/rubble_c9_r4/cameras.json output/rubble_c9_r4_lod/
python viewer.py config/rubble_c9_r4_lod.yaml
```

## 📝 TODO List

- \[x\] First Release.
- \[x\] Release CityGaussian code.
- \[x\] Release ColMap results of main datasets.
- \[x\] Release detailed instruction for custom dataset usage.
- \[ \] Release checkpoints on main datasets.

## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 🤗 Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```latex
@article{liu2024citygaussian,
  title={Citygaussian: Real-time high-quality large-scale scene rendering with gaussians},
  author={Liu, Yang and Guan, He and Luo, Chuanchen and Fan, Lue and Wang, Naiyan and Peng, Junran and Zhang, Zhaoxiang},
  journal={arXiv preprint arXiv:2404.01133},
  year={2024}
}
```

## 👏 Acknowledgements

This repo benefits from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [LightGaussian](https://github.com/VITA-Group/LightGaussian), [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning). Thanks for their great work!
