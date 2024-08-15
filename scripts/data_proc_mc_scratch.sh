# MatrixCity, Aerial View, block All
mkdir data/matrix_city/aerial/train/block_all
mkdir data/matrix_city/aerial/test/block_all_test
mkdir data/matrix_city/aerial/train/block_all/input
mkdir data/matrix_city/aerial/test/block_all_test/input
cp data/matrix_city/aerial/pose/block_all/transforms_train.json data/matrix_city/aerial/train/block_all/transforms.json
cp data/matrix_city/aerial/pose/block_all/transforms_test.json data/matrix_city/aerial/test/block_all_test/transforms.json

python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/train/block_all
python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/test/block_all_test
python convert_cam.py -s data/matrix_city/aerial/train/block_all
python convert_cam.py -s data/matrix_city/aerial/test/block_all_test