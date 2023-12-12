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

