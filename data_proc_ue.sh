SOURCE="street/block2"
SUB="street/block2_5"
SCENE="street"
INTERVAL=5

python tools/convert_json_$SCENE.py --source_path data/UE-collected/$SOURCE --ref_path data/UE-collected/transforms_ref.json --train
python tools/sparse_select_$SCENE.py --dense_path data/UE-collected/$SOURCE --sparse_path data/UE-collected/$SUB --interval $INTERVAL
python tools/transform_json2txt.py --source_path data/UE-collected/$SUB
python convert_cam.py -s data/UE-collected/$SUB
# python convert.py -s data/UE-collected/$SUB