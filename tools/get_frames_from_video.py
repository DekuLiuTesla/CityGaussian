# -*- coding:utf8 -*-
import cv2
import os
import shutil
import sys
from tqdm import tqdm
from argparse import ArgumentParser
 
def get_frame_from_video(video_paths, save_path, interval, start_idx):

    if save_path is None:
        save_path = video_paths[0].split(".")[0] + "_frames/input/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('path of %s is built' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)
    
    start_idx = 0
    for video_path in video_paths:
        
        video_capture = cv2.VideoCapture(video_path)
        
        for i in tqdm(range(0, int(video_capture.get(7)), interval), desc=f"Extracting frames from {video_path}"):
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = video_capture.read()
            if success:
                cv2.imwrite(save_path + str(i+start_idx) + '.jpg', frame)
            else:
                break
        
        start_idx = i + start_idx + interval
 
 
if __name__ == '__main__':
    parser = ArgumentParser(description="Convert each panorama image to 6 perspective images")
    parser.add_argument("--video_paths", type=str, required=True, nargs="+")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--frame_interval", type=int, default=15)
    parser.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    get_frame_from_video(args.video_paths, args.save_path, args.frame_interval, args.start_idx)