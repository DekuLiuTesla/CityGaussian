# -*- coding:utf8 -*-
import cv2
import os
import shutil
import sys
from tqdm import tqdm
from argparse import ArgumentParser
 
def get_frame_from_video(video_path, save_path, interval, start_idx):

    if save_path is None:
        save_path = video_path.split(".")[0] + "_frames/input/"

    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print('path of %s is build' % save_path)
    else:
        shutil.rmtree(save_path)
        os.makedirs(save_path)
        print('path of %s already exist and rebuild' % save_path)
 
    video_capture = cv2.VideoCapture(video_path)
    
    for i in tqdm(range(0, int(video_capture.get(7)), interval)):
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, i)
        success, frame = video_capture.read()
        if success:
            cv2.imwrite(save_path + str(i+start_idx) + '.jpg', frame)
        else:
            break
 
 
if __name__ == '__main__':
    parser = ArgumentParser(description="Convert each panorama image to 6 perspective images")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--frame_interval", type=int, default=15)
    parser.add_argument("--start_idx", type=int, default=0)

    args = parser.parse_args(sys.argv[1:])
    get_frame_from_video(args.video_path, args.save_path, args.frame_interval, args.start_idx)