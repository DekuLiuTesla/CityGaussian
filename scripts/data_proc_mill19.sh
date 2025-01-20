# Mill19, Building, Rubble
ln -s data/mill19/building-pixsfm/train/rgbs data/mill19/building-pixsfm/train/images
ln -s data/mill19/building-pixsfm/val/rgbs data/mill19/building-pixsfm/val/images

ln -s data/mill19/rubble-pixsfm/train/rgbs data/mill19/rubble-pixsfm/train/images
ln -s data/mill19/rubble-pixsfm/val/rgbs data/mill19/rubble-pixsfm/val/images

mv data/colmap_results/building/train/sparse data/mill19/building-pixsfm/train
mv data/colmap_results/building/val/sparse data/mill19/building-pixsfm/val

mv data/colmap_results/rubble/train/sparse data/mill19/rubble-pixsfm/train
mv data/colmap_results/rubble/val/sparse data/mill19/rubble-pixsfm/val

