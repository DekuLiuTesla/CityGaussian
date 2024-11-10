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

The links above points to the papers. Users could follow the [instruction](doc/citygaussianv1_readme.md) to download COLMAP results, checkpoints, and try the V1 of our CityGaussian. The code of V2 is also coming soon.

## 📰 News
**[2024.11.04]** Announcement of our [CityGaussianV2](https://dekuliutesla.github.io/CityGaussianV2/)!

**[2024.10.12]** Checkpoints on main datasets have been released!

**[2024.10.11]** Updates FAQ! If you are stucked, please first check whether it can solves the problem.

**[2024.08.20]** Updates [Custom Dataset Instructions](doc/custom_dataset.md)! 

**[2024.08.05]** Our code is now available! Welcome to try it out!

**[2024.07.18]** Camera Ready version now can be accessed through arXiv. More insights are included.



## 📝 TODO List

- \[ \] Release the V2 of CityGaussian.
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

This repo benefits from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting), [LightGaussian](https://github.com/VITA-Group/LightGaussian), [Gaussian Lightning](https://github.com/yzslab/gaussian-splatting-lightning). Thanks for their great work!

