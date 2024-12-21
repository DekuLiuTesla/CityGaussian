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

