import bpy
import json
import uuid
import os, shutil
import numpy as np
from argparse import ArgumentParser

from cam_pose_utils.cam_reader import readPklSceneInfo
from render_utils.texture_allocator import TextureAllocator
from render_utils.background_generator import draw_background
from generate_video import generate_video

parser = ArgumentParser(description='bpy arg parser')
parser.add_argument("--cuda_id", type=int, default=0)
parser.add_argument("--load_dir", type=str, default='data/mesh_data/mesh_dtu/scan24')
parser.add_argument("--config_dir", type=str, default='render_cfgs/dtu')
parser.add_argument("--mesh_file", type=str, default='mesh.ply')
parser.add_argument("--save_dir", type=str, default='')
parser.add_argument("--is_texture", action='store_true')
parser.add_argument("--image_only", action='store_true')

parser.add_argument('--debug_mode', type=int, default=-1, help='How many images to render (in debug mode). -1 means all images (not in debug mode).')
parser.add_argument('--debug_video_step', type=int, default=1)
parser.add_argument('--write_cover', action='store_true')
parser.add_argument("--fps", type=int, default=60)
parser.add_argument('--fov_scale', type=float, default=1.0)
args = parser.parse_args()

if args.save_dir == '':
    unique_str = str(uuid.uuid4())
    args.save_dir = os.path.join("./output/", unique_str[0:10])

# load config files
cfg_dir = args.config_dir
with open(os.path.join(cfg_dir, "light.json")) as f1:
    light_cfg = json.load(f1)
with open(os.path.join(cfg_dir, "background.json")) as f2:
    back_sphere_radius = json.load(f2)

if args.is_texture:
    light_cfg = light_cfg['texture']
    texture_allocator = TextureAllocator(bpy)
    save_dir = os.path.join(args.save_dir, 'texture')
else:
    light_cfg = light_cfg['mesh']
    save_dir = os.path.join(args.save_dir, 'mesh')

# make save_dir
if os.path.exists(save_dir) and not args.write_cover:
    assert False, 'The save directory already exists.'
os.makedirs(save_dir, exist_ok=True)

# set render configs
bpy.data.objects['Light'].location = light_cfg['pose'][0]
bpy.data.lights['Light'].energy = light_cfg['energy'][0]
bpy.ops.object.light_add(type="POINT", radius=1, location=light_cfg['pose'][1])
bpy.data.objects['Point'].data.energy = light_cfg['energy'][1]
bpy.ops.object.light_add(type="POINT", radius=1, location=light_cfg['pose'][2])
bpy.data.objects['Point.001'].data.energy = light_cfg['energy'][2]

# draw the sphere background
draw_background(args.is_texture)
bpy.data.objects['SurfSphere'].scale = (back_sphere_radius, back_sphere_radius, back_sphere_radius)
bpy.data.objects['SurfSphere'].location = (0.0, 0, 0)
bpy.data.objects['SurfSphere'].rotation_euler = (0, 0, 0)

# set the render configuration for Cycles
bpy.data.scenes['Scene'].render.engine = 'CYCLES'
bpy.data.scenes['Scene'].cycles.samples = 1024
bpy.data.scenes['Scene'].cycles.time_limit = 5
bpy.context.scene.cycles.device = 'GPU'
bpy.context.preferences.addons["cycles"].preferences.get_devices()
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'CUDA'

for idx, d in enumerate(bpy.context.preferences.addons["cycles"].preferences.devices):
    d["use"] = 0
    if d["name"].startswith("NVIDIA") and idx == args.cuda_id:
        d["use"] = 1
        # break

bpy.data.objects.remove(bpy.data.objects['Cube'])

mesh_dir = args.load_dir

bpy.ops.import_mesh.ply(filepath=os.path.join(mesh_dir, args.mesh_file))
if args.is_texture:
    texture_allocator.init_texture()
    texture_allocator.set_texture()

bpy.data.objects['Camera'].rotation_mode = 'QUATERNION'
bpy.data.cameras['Camera'].lens_unit = 'FOV'

cam_infos = readPklSceneInfo(os.path.join(args.load_dir, "traj"))
# render all the image
for cam_id, cam_info in enumerate(cam_infos):
    if args.debug_mode > 0 and cam_id % args.debug_video_step != 0:
        continue

    img_name = cam_info.image_name
    data_rot = cam_info.qvec
    data_loc = cam_info.tvec
    data_intr = cam_info.intr_array
    data_name = img_name + '.png'
    
    for i in range(3):
        bpy.data.objects['Camera'].rotation_quaternion[i] = data_rot[i]
        bpy.data.objects['Camera'].location[i] = data_loc[i]
    bpy.data.objects['Camera'].rotation_quaternion[3] = data_rot[3]

    # bpy.data.cameras['Camera'].lens = data_intr[0]
    bpy.data.cameras['Camera'].angle = data_intr[0] * args.fov_scale
    if data_intr[1] > 1:
        bpy.data.scenes['Scene'].render.pixel_aspect_x = data_intr[1]
    else:
        bpy.data.scenes['Scene'].render.pixel_aspect_x = 1
        bpy.data.scenes['Scene'].render.pixel_aspect_y = 1 / data_intr[1]
    bpy.data.scenes['Scene'].render.pixel_aspect_y = 1
    downsample_rate1 = 1
    bpy.data.scenes['Scene'].render.resolution_x = int(data_intr[2]/downsample_rate1)
    bpy.data.scenes['Scene'].render.resolution_y = int(data_intr[3]/downsample_rate1)

    bpy.ops.render.render() # render an image
    bpy.data.images["Render Result"].save_render(filepath=os.path.join(save_dir, data_name)) # save the image

    if args.debug_mode > 0 and cam_id > args.debug_mode * args.debug_video_step:
        break

# clean up
# bpy.data.objects.remove(bpy.data.objects["mesh"])
bpy.ops.outliner.orphans_purge(do_recursive=True)

# generate_video
if not args.image_only:
    generate_video(path=save_dir, fps=args.fps)