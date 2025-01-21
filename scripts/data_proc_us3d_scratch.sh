# UrbanScene3D, Residence, Sci-Art
python tools/copy_images.py --image_path data/urban_scene_3d/Residence/photos --dataset_path data/urban_scene_3d/residence-pixsfm
python tools/copy_images.py --image_path data/urban_scene_3d/Sci-Art/photos --dataset_path data/urban_scene_3d/sci-art-pixsfm

rm -rf data/urban_scene_3d/residence-pixsfm/train/sparse
rm -rf data/urban_scene_3d/residence-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/urban_scene_3d/residence-pixsfm
python tools/convert_cam.py -s data/urban_scene_3d/residence-pixsfm/train
python tools/convert_cam.py -s data/urban_scene_3d/residence-pixsfm/val

rm -rf data/urban_scene_3d/sci-art-pixsfm/train/sparse
rm -rf data/urban_scene_3d/sci-art-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/urban_scene_3d/sci-art-pixsfm
python tools/convert_cam.py -s data/urban_scene_3d/sci-art-pixsfm/train
python tools/convert_cam.py -s data/urban_scene_3d/sci-art-pixsfm/val
