# Prepare UE-collected data
1. download data, unzip and put them into `path/to/scene/block_i/input` folder, for instance, the images of block_i can be put into `data/UE-collected/aerial/block_i` folder

2. download the `transforms_train.json` that describing the camera parameters of all blocks, then rename it as `transforms_raw.json` and save it to folder as `path/to/scene/block_i/transforms_raw.json`, for instance `data/UE-collected/aerial/block_1/transforms_raw.json`

3. generate acceptable transforms_train.json required by 3DGS using
```bash
python tools/convert_json_aerial.py --source_path path/to/scene/block_i --ref_path path/to/transform_ref.json --train
```

4. (Optional) generate subset of data with appointed interval using
```bash
python tools/sparse_select_aerial.py --dense_path path/to/scene/block_i --sparse_path  path/to/scene/block_i_j --interval j
```

5. (Optional) transform the json file to txt file acceptable for COLMAP
```bash
python tools/transform_json2txt.py --source_path path/to/scene/block_i_j
```

6. (Optional) generate the sparse point cloud using COLMAP
```bash
# if not use existing camera parameters:
python convert.py -s path/to/scene/block_i_j 
# if use existing camera parameters:
python convert_cam.py -s path/to/scene/block_i_j 
```