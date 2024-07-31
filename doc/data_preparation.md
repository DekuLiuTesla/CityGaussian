## Prepare MatrixCity dataset
1. Download small_city version of aerial data of [MatricCity](https://github.com/city-super/MatrixCity) and save to `data/matrix_city/aerial`.

2. Unzip the data to `input` folder for each block in train and test set. You can take `scripts/untar_matrixcity_train.sh` and `scripts/untar_matrixcity_test.sh` as reference. 

3. Run following command to prepare data (The COLMAP step make take a long time for over 5000 images ).
    ```bash
    bash scripts/data_proc_mc.sh
    ``` 


## Prepare Mill19 & UrbanScene3D datasets

1. Download data of Mill19 and UrbanScene3D according to instruction from [MegaNeRF](https://github.com/cmusatyalab/mega-nerf). Save the data to `data/mill19` and `data/urban_scene_3d` respectively.

2. Run following command to prepare data of Mill19 (The COLMAP step make take a long time for large amount of images ).
    ```bash
    bash scripts/data_proc_mill19.sh
    ``` 

3. Run following command to prepare data of UrbanScene3D (The COLMAP step make take a long time for large amount of images ).
    ```bash
    bash scripts/data_proc_us3d.sh
    ```

