# Mill19, Building, Rubble
ln -s data/mill19/building-pixsfm/train/rgbs data/mill19/building-pixsfm/train/input
ln -s data/mill19/building-pixsfm/val/rgbs data/mill19/building-pixsfm/val/input

ln -s data/mill19/rubble-pixsfm/train/rgbs data/mill19/rubble-pixsfm/train/input
ln -s data/mill19/rubble-pixsfm/val/rgbs data/mill19/rubble-pixsfm/val/input

rm -rf data/mill19/building-pixsfm/train/sparse
rm -rf data/mill19/building-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/mill19/building-pixsfm
python tools/convert_cam.py -s data/mill19/building-pixsfm/train
python tools/convert_cam.py -s data/mill19/building-pixsfm/val

rm -rf data/mill19/rubble-pixsfm/train/sparse
rm -rf data/mill19/rubble-pixsfm/val/sparse
python tools/transform_pt2txt.py --source_path data/mill19/rubble-pixsfm
python tools/convert_cam.py -s data/mill19/rubble-pixsfm/train
python tools/convert_cam.py -s data/mill19/rubble-pixsfm/val

