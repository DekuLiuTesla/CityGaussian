# Aerial View, for block 1 to block 10
for num in {1..10}  
do  
python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial/train/block_$num
python convert_cam.py -s data/matrix_city/aerial/train/block_$num
done 

for num in {1..10}  
do  
python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial/test/block_${num}_test
python convert_cam.py -s data/matrix_city/aerial/test/block_${num}_test
done 

# Aerial View, block A
mkdir data/matrix_city/aerial/train/block_A
mkdir data/matrix_city/aerial/test/block_A_test
mkdir data/matrix_city/aerial/train/block_A/input
mkdir data/matrix_city/aerial/test/block_A_test/input
cp data/matrix_city/aerial/pose/block_A/transforms_train.json data/matrix_city/aerial/train/block_A/transforms.json
cp data/matrix_city/aerial/pose/block_A/transforms_test.json data/matrix_city/aerial/test/block_A_test/transforms.json

python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/train/block_A
python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/test/block_A_test
python convert_cam.py -s data/matrix_city/aerial/train/block_A
python convert_cam.py -s data/matrix_city/aerial/test/block_A_test

# Aerial View, block B
mkdir data/matrix_city/aerial/train/block_B
mkdir data/matrix_city/aerial/test/block_B_test
mkdir data/matrix_city/aerial/train/block_B/input
mkdir data/matrix_city/aerial/test/block_B_test/input
cp data/matrix_city/aerial/pose/block_B/transforms_train.json data/matrix_city/aerial/train/block_B/transforms.json
cp data/matrix_city/aerial/pose/block_B/transforms_test.json data/matrix_city/aerial/test/block_B_test/transforms.json

python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/train/block_B
python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/test/block_B_test
python convert_cam.py -s data/matrix_city/aerial/train/block_B
python convert_cam.py -s data/matrix_city/aerial/test/block_B_test

# Aerial View, block C
mkdir data/matrix_city/aerial/train/block_C
mkdir data/matrix_city/aerial/test/block_C_test
mkdir data/matrix_city/aerial/train/block_C/input
mkdir data/matrix_city/aerial/test/block_C_test/input
cp data/matrix_city/aerial/pose/block_C/transforms_train.json data/matrix_city/aerial/train/block_C/transforms.json
cp data/matrix_city/aerial/pose/block_C/transforms_test.json data/matrix_city/aerial/test/block_C_test/transforms.json

python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/train/block_C
python tools/transform_json2txt_mc_all.py --source_path data/matrix_city/aerial/test/block_C_test
python convert_cam.py -s data/matrix_city/aerial/train/block_C
python convert_cam.py -s data/matrix_city/aerial/test/block_C_test

# Aerial View, block All
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

# Street View
mkdir data/matrix_city/street/train/block_A
mkdir data/matrix_city/street/test/block_A_test
python tools/transform_json2txt_mc_all_street.py --source_path data/matrix_city/street/train/block_A --intrinsic_path data/matrix_city/street/pose/block_A/transforms_train.json
python tools/transform_json2txt_mc_all_street.py --source_path data/matrix_city/street/test/block_A_test --intrinsic_path data/matrix_city/street/pose/block_A/transforms_test.json
python convert_cam.py -s data/matrix_city/street/train/block_A
python convert_cam.py -s data/matrix_city/street/test/block_A_test

mkdir data/matrix_city/street/train/block_small
mkdir data/matrix_city/street/test/block_small_test
python tools/transform_json2txt_mc_all_street.py --source_path data/matrix_city/street/train/block_small --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
python tools/transform_json2txt_mc_all_street.py --source_path data/matrix_city/street/test/block_small_test --intrinsic_path data/matrix_city/street/pose/block_small/transforms_test.json
python convert_cam.py -s data/matrix_city/street/train/block_small
python convert_cam.py -s data/matrix_city/street/test/block_small_test