# Prepare MatrixCity dataset
Download dataset: <a href="https://city-super.github.io/matrixcity/">MatrixCity</a>

## Example Structure

The key is that `rgb` and `depth` directories need to be placed in the same directory as `transforms.json`.

```bash
├── small_city/aerial
    ├── block_1
        ├── depth  # provided in the tar files here: https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city_depth
            ├── 0000.exr
            ├── 0001.exr
            ...
        ├── rgb  # provided in the tar files here: https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city
            ├── 0000.png
            ├── 0001.png
            ...
        ├── transforms.json  # Poses of images after removing the images that look outside the map boundary, which are used for training and testing
        ├── transforms_origin.json  # Poses of all original collected images
    ├── block_1_test
        ...  # same as `block_1` above
    ├── block_2
        ...  # same as `block_1` above
    ...
├── small_city/street
    ...  # same as `small_city/aerial` above
```

Please note that the `transforms.json` and `transforms_origin.json` are not the files in `pose` directories. They are located in the sub-directories of the directory where RGB tarball files placed. For example, <a href="https://huggingface.co/datasets/BoDai/MatrixCity/tree/main/small_city/aerial/train/block_1">here are the `json` files for `small_city/aerial/block_1`</a>.

## Usage example
<b>[NOTE]</b> It takes some time to generate a point cloud the first time it runs

* Via command
    ```bash 
    python main.py fit \
        --data.path data/MatrixCity/small_city/aerial \
        --data.parser MatrixCity \
        --data.parser.train '["block_1/transforms.json", "block_2/transforms.json"]' \
        --data.parser.test '["block_1_test/transforms.json", "block_2_test/transforms.json"]' \
        ...
    ```

    The `--data.parser.train` and `--data.parser.test` specify the json files of the blocks you want to use.
* Via config file
  ```bash
  python main.py fit \
      --config configs/matrixcity/gsplat-aerial.yaml \
      --data.path data/MatrixCity/small_city/aerial \
      ...
  ```
  See <a href="https://github.com/yzslab/gaussian-splatting-lightning/tree/main/configs/matrixcity/gsplat-aerial.yaml">`configs/matrixcity/gsplat-aerial.yaml`</a> for more details.
* Mixing aerial and street views is also possible
  ```bash
  python main.py fit \
      --config configs/matrixcity/gsplat-aerial_street-example.yaml \
      --data.path data/MatrixCity/small_city \
      ...
  ```
  
  Please note that the value of `--data.path` is different from above. Take a look <a href="https://github.com/yzslab/gaussian-splatting-lightning/tree/main/configs/matrixcity/gsplat-aerial_street-example.yaml">`configs/matrixcity/gsplat-aerial_street-example.yaml`</a> for more details.
