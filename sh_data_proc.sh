# python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial_street_fusion/aerial
# python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/aerial

# python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial_street_fusion/street/train --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
# python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/street/train

# python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial_street_fusion/street/test --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
# python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/street/test

python tools/transform_json2txt_mc_part.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_train.json --target_path data/matrix_city/aerial_street_fusion/aerial/train
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/aerial/train
python tools/transform_json2txt_mc_part.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_test.json --target_path data/matrix_city/aerial_street_fusion/aerial/test
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/aerial/test

python tools/transform_json2txt_mc_part.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_train.json --target_path data/matrix_city/aerial_street_fusion/street/train --mode street --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/street/train
python tools/transform_json2txt_mc_part.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_test.json --target_path data/matrix_city/aerial_street_fusion/street/test --mode street --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/street/test

mkdir data/matrix_city/aerial_street_fusion/fuse
mkdir data/matrix_city/aerial_street_fusion/fuse/train
mkdir data/matrix_city/aerial_street_fusion/fuse/test

python tools/transform_json2txt_mc_fuse.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_train.json --target_path data/matrix_city/aerial_street_fusion
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/fuse/train

python tools/transform_json2txt_mc_fuse.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_test.json --target_path data/matrix_city/aerial_street_fusion
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/fuse/test