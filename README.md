<br>
<p align="center">
<h1 align="center"><strong>CityGaussian Series for High-quality Large-Scale Scene Reconstruction with Gaussians</strong></h1>
  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/%F0%9F%9A%80%20Project-V1-green)](https://dekuliutesla.github.io/citygs/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-green)](https://huggingface.co/TeslaYang123/CityGaussian)
[![](https://img.shields.io/badge/📄中文解读-V1-green)](https://hub.baai.ac.cn/view/41840)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20Project-V2-blue)](https://dekuliutesla.github.io/CityGaussianV2/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-blue)](https://huggingface.co/TeslaYang123/CityGaussianV2)
[![](https://img.shields.io/badge/📄中文解读-V2-blue)](https://www.jiqizhixin.com/articles/2025-02-05-5)
![GitHub Repo stars](https://img.shields.io/github/stars/DekuLiuTesla/CityGaussian)
[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)


</div>
<p align="center">
  <img width="460" height="300" src="assets/demo.gif">
</p>

This repo contains official implementations of our series of work in large-scale scene reconstruction with Gaussian Splatting, Star ⭐ us if you like it!
- [CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes](https://arxiv.org/pdf/2411.00771) (ICLR 2025)
- [CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians](https://arxiv.org/pdf/2404.01133) (ECCV 2024)

The links above points to the papers. The main branch now has been rebased to [Gaussian Lightning v0.10.1](https://github.com/yzslab/gaussian-splatting-lightning). Feel free to explore the repository!

## 👏 Features
* CityGaussian-style multi-gpu reconstruction with controllable memory cost and no limit on GPU amount
* Analysis of model partition and data assignment
* 2DGS-style mesh extraction & Large-scale scene geometric performance evaluation
* Trajectory aligned rendering & mesh video generation with floater removement
* Features of [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning), including web viewer, MipSplatting, AbsGS, StopThePop, etc.

<details>
<summary><span style="font-weight: bold;">Table Results & Checkpoints </span></summary>

| Scene | SSIM↑ | PSNR↑ | LPIPS↓ | Precision↑ | Recall↑ | F1-Score↑ | #GS(M) |
| :---: | :---: | :---: | :---: | :---:|:---:| :---: | :---: |
| LFLS | 0.744 | 23.44 | 0.246 | 0.556 | 0.400 | 0.466 | 8.19 |
| SMBU | 0.794 | 24.00 | 0.185 | 0.559 | 0.523 | 0.541 | 5.33 |
| Upper Campus | 0.779 | 25.78 | 0.186 | 0.654 | 0.394 | 0.491 | 7.87 |
| MatrixCity Aerial | 0.859 | 27.26 | 0.175 | 0.432 | 0.790 | 0.559 | 8.57 |
| MatrixCity Street | 0.791 | 22.32 | 0.344 | 0.325 | 0.797 | 0.461 | 7.40 |

Note for street view, the F1-Score is lower than that reported in paper, since we sacrifice precision for a better recall and more complete road surface. If unbroken road is prefered, you can adjust `depth_ratio` to 0.0, but surface reconstruction performance will be worse. The checkpoints of CityGSV2 can be found here:

- Baidu Netdisk: https://pan.baidu.com/s/1tRKiJzMLk2-zoyvoa9bkqA?pwd=1b4r
- Hugging Face: https://huggingface.co/TeslaYang123/CityGaussianV2

</details>

## 📰 News
**[2025.01.31]** Checkpoints of CityGaussian V2 has been released!

**[2025.01.22]** CityGaussian V2 has been accepted by ICLR 2025!

**[2025.01.22]** Code of CityGaussian V2 is now released. Welcome to try it out!

**[2024.11.04]** Announcement of our [CityGaussianV2](https://dekuliutesla.github.io/CityGaussianV2/)!

**[2024.10.12]** Checkpoints of V1 on main datasets have been released! 

**[2024.08.05]** Code of CityGaussian V1 is available!

## 🛠 Getting Started
- [Installation](doc/installation.md)
- [Data Preparation](doc/data_preparation.md)
- [Run and Eval](doc/run&eval.md)
- [Video Rendering on GS and Mesh](doc/render_video.md)


## 📝 TODO List

- \[ \] Official Implementation of Appearance Embedding.
- \[ \] Support of V1 style LoD.
- \[x\] Release the checkpoint of CityGaussian V2.
- \[x\] Release the V2 of CityGaussian.
- \[x\] Release code and checkpoints of CityGaussian.
- \[x\] Release ColMap results of main datasets.


## 📄 License

<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## 🤗 Citation
If you find this repository useful, please use the following BibTeX entry for citation.
```latex
@misc{liu2024citygaussianv2efficientgeometricallyaccurate,
      title={CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes}, 
      author={Yang Liu and Chuanchen Luo and Zhongkai Mao and Junran Peng and Zhaoxiang Zhang},
      year={2024},
      eprint={2411.00771},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2411.00771}, 
}
```

```latex
@inproceedings{liu2025citygaussian,
  title={Citygaussian: Real-time high-quality large-scale scene rendering with gaussians},
  author={Liu, Yang and Luo, Chuanchen and Fan, Lue and Wang, Naiyan and Peng, Junran and Zhang, Zhaoxiang},
  booktitle={European Conference on Computer Vision},
  pages={265--282},
  year={2025},
  organization={Springer}
}
```

## 👏 Acknowledgements

This repo benefits from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [2DGS](), [TrimGS](https://github.com/YuxueYang1204/TrimGS), [LightGaussian](https://github.com/VITA-Group/LightGaussian), [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning). Thanks for their great work!

## ❓ FAQ
- _Out of memory occurs in training._ To finish training with limited VRAM, downsampling images or adjusting max_cache_num (we used a rather large 1024) in train_large.py can be a useful practice. Besides, you can increase `prune_ratio` in parallel tuning to further reduce memory cost.

- _Generation of COLMAP results._ We use the ground-truth poses offered by datasets and separately match the train and test sets. And this will be faster and more robust than match from scratch. But indeed it still costs a lot of time.

- _Most blocks are not trained._ The main reason here is the data assigned to most blocks are too few (<50), and to prevent overfitting these blocks won't get trained. This can be attributed to unreasonable aabb setting, please try to adjust it and see if things work.
