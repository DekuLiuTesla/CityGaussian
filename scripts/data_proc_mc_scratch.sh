# MatrixCity, Aerial View, block All
mkdir data/matrix_city/aerial/train/block_all
mkdir data/matrix_city/aerial/test/block_all_test
mkdir data/matrix_city/aerial/train/block_all/input
mkdir data/matrix_city/aerial/test/block_all_test/input
cp data/matrix_city/aerial/pose/block_all/transforms_train.json data/matrix_city/aerial/train/block_all/transforms.json
cp data/matrix_city/aerial/pose/block_all/transforms_test.json data/matrix_city/aerial/test/block_all_test/transforms.json

python tools/transform_json2txt_mc_aerial.py --source_path data/matrix_city/aerial/train/block_all
python tools/transform_json2txt_mc_aerial.py --source_path data/matrix_city/aerial/test/block_all_test
python convert_cam.py -s data/matrix_city/aerial/train/block_all
python convert_cam.py -s data/matrix_city/aerial/test/block_all_test

# Street View
mkdir data/matrix_city/street/train/block_A
mkdir data/matrix_city/street/test/block_A_test
mkdir data/matrix_city/street/train/block_A/input
mkdir data/matrix_city/street/test/block_A_test/input
cp data/matrix_city/street/pose/block_A/transforms_train.json data/matrix_city/street/train/block_A/transforms.json
cp data/matrix_city/street/pose/block_A/transforms_test.json data/matrix_city/street/test/block_A_test/transforms.json

python tools/transform_json2txt_mc_street.py --source_path data/matrix_city/street/train/block_A --intrinsic_path data/matrix_city/street/pose/block_A/transforms_train.json
python tools/transform_json2txt_mc_street.py --source_path data/matrix_city/street/test/block_A_test --intrinsic_path data/matrix_city/street/pose/block_A/transforms_test.json
python tools/convert_cam.py -s data/matrix_city/street/train/block_A
python tools/convert_cam.py -s data/matrix_city/street/test/block_A_test