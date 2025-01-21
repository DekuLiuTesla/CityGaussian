## Render Video
### A. Generate trajectory, filter out floaters, and render a video
```bash
python tools/render_traj.py --output_path outputs/$NAME --filter --train 
```
The script will generate and save a ellipse trajectory according to the training cameras. `--filter` means filter out floaters in each bev pillar according to spatial distribution, and `--train` means use training cameras to generate trajectory. Please refer to the script for more control options.

### B. Render mesh on appointed trajectory with Blender
First, follow the instrution [here](blender/README.md) to install the blender environment. For video rendering, use the following command:
```bash
cd blender
python render_run.py --load_path <path to the mesh>  --traj_path <path to the trajectory> --config_dir <path of the render config>
```
By setting `traj_path`, you can apply the trajectory generated in the previous step. The `config_dir` for GauU-Scene and MatrixCity are provided in `blender/render_cfgs`.