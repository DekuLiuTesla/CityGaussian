## Installation
### A. Clone repository

```bash
# clone repository
git clone https://github.com/DekuLiuTesla/CityGaussian.git
cd CityGaussian
```

### B. Create virtual environment

```bash
# create virtual environment
conda create -yn gspl python=3.9 pip
conda activate gspl
```

### C. Install PyTorch
* Tested on `PyTorch==2.0.1`
* You must install the one match to the version of your nvcc (nvcc --version)
* For CUDA 11.8

  ```bash
  pip install -r requirements/pyt201_cu118.txt
  ```

### D. Install requirements

```bash
pip install -r requirements.txt
```

### E. Install additional package for CityGaussian

```bash
pip install -r requirements/CityGS.txt
```
Note that here we use modified version of Trim2DGS rasterizer, so as to resolve [impulse noise problem](https://github.com/hbb1/2d-gaussian-splatting/issues/174) under street views. This version also avoids interference from out-of-view surfels.