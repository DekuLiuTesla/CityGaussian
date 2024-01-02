# 3DGS
```bash
pip install transforms3d
pip install lightning
pip install wandb
pip install torch-scatter==2.1.0+pt112cu116
conda install pytorch3d -c pytorch3d
```

# COLMAP
a. sudo dpkg -i cuda-keyring_1.0-1_all.deb

b. rm /etc/apt/sources.list.d/cuda.list

c. add the following source /etc/apt/sources.list and comment others (key concept is to ensure all the following packages can be found on https://packages.ubuntu.com/. For instance, the following source appoints distribution "focal" for Ubuntu 20.04LTS, then all packages should be able to be found by searching corresponding keyword while setting Distribution to "focal".)
```
deb https://artifactory.tusimple.ai/artifactory/ubuntu/ focal main restricted universe multiverse
```

d. sudo apt-get update

e. install dependencies
```
sudo apt-get install \
    git \
    cmake \
    ninja-build \
    build-essential \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libeigen3-dev \
    libflann-dev \
    libfreeimage-dev \
    libmetis-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libsqlite3-dev \
    libglew-dev \
    qtbase5-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    libceres-dev
```

g. Install colmap
```
git clone https://github.com/colmap/colmap.git
mkdir build
cd build
cmake .. -GNinja
ninja
sudo ninja install
```

h. If meet error -- unsupported GNU version gcc later than 10 are not supported
```
sudo apt-get install gcc-10
sudo apt-get install g++-10
rm /usr/bin/gcc 
sudo ln -s /usr/bin/gcc-10 /usr/bin/gcc
rm /usr/bin/g++
sudo ln -s /usr/bin/g++-10 /usr/bin/g++
```

f. If meet error :
```bash
CMake Error at cmake/FindDependencies.cmake:125 (message):
  You must set CMAKE_CUDA_ARCHITECTURES to e.g.  'native', 'all-major', '70',
  etc.  More information at
  https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
Call Stack (most recent call first):
  CMakeLists.txt:86 (include)
```
Then add `set(CMAKE_CUDA_ARCHITECTURES 70 75)` to colmap/cmake/FindDependencies.cmake

e. If meet warning when use `cmake .. -GNinja`:
```
runtime library [libmpfr.so.6] in /usr/lib/x86_64-linux-gnu may be hidden by files in:
  /home/rodrigo/anaconda3/lib
runtime library [libgmp.so.10] in /usr/lib/x86_64-linux-gnu may be hidden by files in:
  /home/rodrigo/anaconda3/lib
```
Then temporarily mask the anaconda, modify the name of the anaconda folder, and modify it after the installation is complete. To be specific:
```
mv /root/anaconda3 /root/axx
cmake .. -GNinja
ninja
sudo ninja install
mv /root/axx /root/anaconda3 
```
