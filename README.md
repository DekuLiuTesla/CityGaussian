<br>
<p align="center">
<h1 align="center"><strong>CityGaussian Series for High-quality Large-Scale Scene Reconstruction with Gaussians</strong></h1>
  <p align="center">
    <em>Institute of Automation, Chinese Academy of Sciences; University of Chinese Academy of Sciences</em>
  </p>
</p>

<div id="top" align="center">

[![](https://img.shields.io/badge/%F0%9F%9A%80%20Project-V1-green)](https://dekuliutesla.github.io/citygs/)
[![](https://img.shields.io/badge/%F0%9F%9A%80%20Project-V2-blue)](https://dekuliutesla.github.io/CityGaussianV2/)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-Hugging%20Face-orange)](https://huggingface.co/TeslaYang123/CityGaussian)
![GitHub Repo stars](https://img.shields.io/github/stars/DekuLiuTesla/CityGaussian)

</div>

This repo contains official implementations of our series of work in large-scale scene reconstruction with Gaussian Splatting, Star ⭐ us if you like it!
- [CityGaussianV2: Efficient and Geometrically Accurate Reconstruction for Large-Scale Scenes](https://arxiv.org/pdf/2411.00771)
- [CityGaussian: Real-time High-quality Large-Scale Scene Rendering with Gaussians](https://arxiv.org/pdf/2404.01133) (ECCV 2024)

The links above points to the papers. The main branch now has been rebased to [Gaussian Lightning v0.10.1](https://github.com/yzslab/gaussian-splatting-lightning). Feel free to explore the repository!

## 👏 Features
* CityGaussian-style multi-gpu Gaussian Splatting training with controllable memory cost and no limit on GPU amount
* Analysis of model partition and data assignment
* Floater removement and 2DGS-style mesh extraction
* Large-scale scene geometric performance evaluation
* Trajectory aligned rendering & mesh video generation
* Features of [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning), including web viewer, MipSplatting, AbsGS, StopThePop, etc.


## 📰 News
**[2025.01.22]** Code of CityGaussian V2 is now released. Welcome to try it out!

**[2024.11.04]** Announcement of our [CityGaussianV2](https://dekuliutesla.github.io/CityGaussianV2/)!

**[2024.10.12]** Checkpoints of V1 on main datasets have been released! 

**[2024.08.05]** Code of CityGaussian V1 is available!

## 🛠 Getting Started
- [Installation](doc/installation.md)
- [Data Preparation](doc/data_preparation.md)
- [Run and Eval](doc/run&eval.md)
- [Video Rendering on GS and Mesh](doc/run&eval.md)


## 📝 TODO List

- \[ \] Official Implementation of Appearance Embedding.
- \[ \] Support of V1 style LoD.
- \[ \] Release the checkpoint of CityGaussian V2.
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

