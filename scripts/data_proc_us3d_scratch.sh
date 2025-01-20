# UrbanScene3D, Residence, Sci-Art
python copy_images.py --image_path data/urban_scene_3d/Residence/photos --dataset_path data/urban_scene_3d/residence-pixsfm
python copy_images.py --image_path data/urban_scene_3d/Sci-Art/photos --dataset_path data/urban_scene_3d/sci-art-pixsfm

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
