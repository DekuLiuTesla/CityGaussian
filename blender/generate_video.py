import imageio
import os
from tqdm import tqdm
from argparse import ArgumentParser

def generate_video(path, fps):
    # if not is_texture:
    #     write_dir = os.path.join(path, 'videos', 'mesh')
    #     load_dir = os.path.join(path, 'mesh')
    # else:
    #     write_dir = os.path.join(path, 'videos', 'texture')
    #     load_dir = os.path.join(path, 'texture')
    load_dir = path
    if not os.path.isdir(load_dir):
        assert False
    write_dir = os.path.join(path, 'videos')
    if not os.path.exists(write_dir):
        os.makedirs(write_dir)
    video = imageio.get_writer(os.path.join(write_dir, 'video.mp4'), fps=fps)
    image_list = sorted(os.listdir(load_dir))
    for i in tqdm(range(len(image_list)), desc=f"Creating video"):
        path = os.path.join(load_dir, image_list[i])
        if os.path.isdir(path):
            continue
        image = imageio.imread(path)
        video.append_data(image)
    video.close()

if __name__ == "__main__":

    parser = ArgumentParser(description='video generator arg parser')
    parser.add_argument('--load_dir', type=str, default="render_res")
    parser.add_argument("--is_texture", action="store_true")
    parser.add_argument("--fps", type=int, default=60)
    args = parser.parse_args()
    generate_video(path=args.load_dir, fps=args.fps)