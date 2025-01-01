# PolyCam's raw data

* Only the raw data in LiDAR or Room mode are supported

* Take a look <a href="https://docs.nerf.studio/quickstart/custom_dataset.html#polycam-capture">NeRFStudio's quick start</a> about how to export raw data

## Using the raw data in this repo.
1. Unzip the raw data file

2. Convert it to NGP's `transforms.json` and generate a point cloud from the depth maps
    ```bash
    python utils/polycam2ngp.py RAW_DATA_DIR
    ```
    The `RAW_DATA_DIR` must contain the directory named `keyframes`.

3. Start training
    ```bash
    python main.py fit \
        --config config/gsplat.yaml \
        --data.path RAW_DATA_DIR \
        --data.parser internal.dataparsers.ngp_dataparser.NGP \
        ...
    ```
