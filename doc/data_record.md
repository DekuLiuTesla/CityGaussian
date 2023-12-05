# Prepare UE-collected data
1. download data, unzip and put them into `path/to/scene/block_i/input` folder, for instance, the images of block_i can be put into `data/UE-collected/aerial/block_i` folder

2. download the `transforms_train.json` that describing the camera parameters of all blocks, then rename it as `transforms_raw.json` and save it to folder as `path/to/scene/block_i/transforms_raw.json`, for instance `data/UE-collected/aerial/block_1/transforms_raw.json`

3. adjusted `SCENE` in `data.sh` to "street" or "aerial", according to the scene you are going to use

4. If you don't need to seperate a sparse version of data with specified interval, then just comments the line 7 in `data.sh`

5. If you want to run COLMAP without appointed camera parameters, then replace line 9 in data.sh with
```bash
python convert.py -s data/UE-collected/$SUB
```