# python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial_street_fusion/aerial
# python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/aerial

python tools/transform_json2txt_mc_part.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_train.json --target_path data/matrix_city/aerial_street_fusion/aerial/train
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/aerial/train
python tools/transform_json2txt_mc_part.py --transforms_path data/matrix_city/aerial_street_fusion/pose/transforms_test.json --target_path data/matrix_city/aerial_street_fusion/aerial/test
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/aerial/test

python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial_street_fusion/street/train --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/street/train

python tools/transform_json2txt_mc.py --source_path data/matrix_city/aerial_street_fusion/street/test --intrinsic_path data/matrix_city/street/pose/block_small/transforms_train.json
python tools/convert_cam.py -s data/matrix_city/aerial_street_fusion/street/test