python copy_images.py --image_path data/urban_scene_3d/Campus/photos --dataset_path data/urban_scene_3d/campus-pixsfm
python copy_images.py --image_path data/urban_scene_3d/Residence/photos --dataset_path data/urban_scene_3d/residence-pixsfm
python copy_images.py --image_path data/urban_scene_3d/Sci-Art/photos --dataset_path data/urban_scene_3d/sci-art-pixsfm

ln -s /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/campus-pixsfm/train/rgbs /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/campus-pixsfm/train/input
ln -s /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/campus-pixsfm/val/rgbs /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/campus-pixsfm/val/input

ln -s /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/residence-pixsfm/train/rgbs /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/residence-pixsfm/train/input
ln -s /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/residence-pixsfm/val/rgbs /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/residence-pixsfm/val/input

ln -s /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/sci-art-pixsfm/train/rgbs /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/sci-art-pixsfm/train/input
ln -s /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/sci-art-pixsfm/val/rgbs /home/yang_liu/python_workspace/3DGS/data/urban_scene_3d/sci-art-pixsfm/val/input

ln -s /home/yang_liu/python_workspace/3DGS/data/mill19/building-pixsfm/train/rgbs /home/yang_liu/python_workspace/3DGS/data/mill19/building-pixsfm/train/input
ln -s /home/yang_liu/python_workspace/3DGS/data/mill19/building-pixsfm/val/rgbs /home/yang_liu/python_workspace/3DGS/data/mill19/building-pixsfm/val/input

ln -s /home/yang_liu/python_workspace/3DGS/data/mill19/rubble-pixsfm/train/rgbs /home/yang_liu/python_workspace/3DGS/data/mill19/rubble-pixsfm/train/input
ln -s /home/yang_liu/python_workspace/3DGS/data/mill19/rubble-pixsfm/val/rgbs /home/yang_liu/python_workspace/3DGS/data/mill19/rubble-pixsfm/val/input

rm -rf data/urban_scene_3d/campus-pixsfm/train/sparse
rm -rf data/urban_scene_3d/campus-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/urban_scene_3d/campus-pixsfm
python convert_cam.py -s data/urban_scene_3d/campus-pixsfm/train
python convert_cam.py -s data/urban_scene_3d/campus-pixsfm/val

rm -rf data/urban_scene_3d/residence-pixsfm/train/sparse
rm -rf data/urban_scene_3d/residence-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/urban_scene_3d/residence-pixsfm
python convert_cam.py -s data/urban_scene_3d/residence-pixsfm/train
python convert_cam.py -s data/urban_scene_3d/residence-pixsfm/val

rm -rf data/urban_scene_3d/sci-art-pixsfm/train/sparse
rm -rf data/urban_scene_3d/sci-art-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/urban_scene_3d/sci-art-pixsfm
python convert_cam.py -s data/urban_scene_3d/sci-art-pixsfm/train
python convert_cam.py -s data/urban_scene_3d/sci-art-pixsfm/val

rm -rf data/mill19/building-pixsfm/train/sparse
rm -rf data/mill19/building-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/mill19/building-pixsfm
python convert_cam.py -s data/mill19/building-pixsfm/train
python convert_cam.py -s data/mill19/building-pixsfm/val

rm -rf data/mill19/rubble-pixsfm/train/sparse
rm -rf data/mill19/rubble-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/mill19/rubble-pixsfm
python convert_cam.py -s data/mill19/rubble-pixsfm/train
python convert_cam.py -s data/mill19/rubble-pixsfm/val