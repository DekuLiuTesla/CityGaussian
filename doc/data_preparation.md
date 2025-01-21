## Download COLMAP Results
For COLMAP, we recommend to directly use our generated results:

- **Google Drive**: https://drive.google.com/file/d/1Uz1pSTIpkagTml2jzkkzJ_rglS_z34p7/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1zX34zftxj07dCM1x5bzmbA?pwd=1t6r

Suppose that you have downloaded and unzip the COLMAP results to `./data` folder like
```
├── data
│   ├── colmap_results
│   │   ├── matrix_city_aerial
│   │   │   ├── train
│   │   │   │   ├── sparse
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   ├── test
│   │   │   │   ├── sparse
│   │   │   │   │   ├── 0
│   │   │   │   │   │   ├── cameras.bin
│   │   │   │   │   │   ├── images.bin
│   │   │   │   │   │   ├── points3D.bin
│   │   ├── matrix_city_street
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── building
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── residence
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── rubble
│   │   │   ├── train
│   │   │   ├── val
│   │   ├── sciart
│   │   │   ├── train
│   │   │   ├── val

```

## Prepare for Tanks and Temple style geometry evaluation
To evaluate surface reconstruction accuracy, please first down load the ground truth point cloud (.ply) and crop volume file (.json) to the `./data`. The transform.txt is used to coarsely align target points to gt points. The links are:
- **Google Drive**: https://drive.google.com/file/d/18L9AEJS2SNva7JgL2-DmqhoNtfDPSSY5/view?usp=sharing
- **Baidu Netdisk**: https://pan.baidu.com/s/1WBJkj42AOsgrNb7YBmcbGg?pwd=in4i

**Note that Mill19 and UrbanScene3D doesn't provide ground-truth point cloud, thus they are not included.** In `./scripts/gt_generate.sh`, we provide the script about how we downsample the ground truth point cloud and generate the crop volume. If you need to process the custom dataset, please refer to the script.

## Prepare GauU-Scene dataset
For GauU-Scene dataset, please follow instruction [here](https://saliteta.github.io/CUHKSZ_SMBU/) to download. The data includes RGB images and COLMAP results.

## Prepare MatrixCity dataset
1. Download small_city version of aerial data of [MatricCity](https://github.com/city-super/MatrixCity) and save to `data/matrix_city/aerial`.

2. Unzip the data to `input` folder for each block in train and test set. You can take `scripts/untar_matrixcity_train.sh` and `scripts/untar_matrixcity_test.sh` as reference. 

3. Run following command to prepare data with generated COLMAP results.
    ```bash
    bash scripts/data_proc_mc.sh
    ``` 

4. [Optional] Run following command to prepare data from scratch (The COLMAP step make take a long time for over 5000 images ).
    ```bash
    bash scripts/data_proc_mc_scratch.sh
    ``` 

5. The process above also applies to the street view of MatrixCity's small_city version. The mentioned scripts also contains the required steps for street view data preparation.

## Prepare Mill19 & UrbanScene3D datasets

1. Download data of Mill19 and UrbanScene3D according to instruction from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf). Save the data to `data/mill19` and `data/urban_scene_3d` respectively. It is worth noticing that the used UrbanScene3D-V1 should be downloaded from [here](https://github.com/Linxius/UrbanScene3D).

2. Run following command to prepare data of Mill19 with generated COLMAP results.
    ```bash
    bash scripts/data_proc_mill19.sh
    ``` 

3. [Optional] Run following command to prepare data of Mill19 from scratch (The COLMAP step make take a long time for large amount of images ).
    ```bash
    bash scripts/data_proc_mill19_scratch.sh
    ``` 

4. Run following command to prepare data of UrbanScene3D with generated COLMAP results.
    ```bash
    bash scripts/data_proc_us3d.sh
    ```

5. [Optional] Run following command to prepare data of UrbanScene3D from scratch (The COLMAP step make take a long time for large amount of images ).
    ```bash
    bash scripts/data_proc_us3d_scratch.sh
    ```

###  Data Preprocessing

The desried dataset folder structure is:
```
├── data
│   ├── your_scene
│   │   ├── images
│   │   ├── sparse
│   │   │   ├── 0
│   │   │   │   ├── cameras.bin
│   │   │   │   ├── points3D.bin
│   │   │   │   ├── images.bin
│   ├── geometry_gt
│   │   ├── your_scene
│   │   │   ├── your_gt_pcd.ply
│   │   │   ├── your_gt_pcd.json
│   │   │   ├── transform.txt [optional]
```

Firstly, downsample the images to desired size:
```bash
python utils/image_downsample.py data/your_scene/images --factor $DOWNSAMPLE_RATIO
```
The $DOWNSAMPLE_RATIO is 3.4175 for GauU-Scene, 1.2 for aerial view of MatrixCity, 1.0 for street view of MatrixCity (no downsample), and 4.0 for Mill19 and UrbanScene3D.

Secondly, prepare [Depth Anything V2](https://depth-anything-v2.github.io/) for depth regularization:
```bash
# clone the repo.
git clone https://github.com/DepthAnything/Depth-Anything-V2 utils/Depth-Anything-V2

# NOTE: do not run `pip install -r utils/Depth-Anything-V2/requirements.txt`

# download the pretrained model `Depth-Anything-V2-Large`
mkdir utils/Depth-Anything-V2/checkpoints
wget -O utils/Depth-Anything-V2/checkpoints/depth_anything_v2_vitl.pth "https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true"
```
The depth can be generated with:
```bash
python utils/estimate_dataset_depths.py data/your_scene -d $DOWNSAMPLE_RATIO
```
