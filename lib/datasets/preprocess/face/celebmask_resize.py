#!/usr/bin/python
# -*- encoding: utf-8 -*-
# Reference: https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py
#            

# other resource: 
#                   https://github.com/switchablenorms/CelebAMask-HQ
#                   https://github.com/zllrunning/face-parsing.PyTorch
#                   https://github.com/JACKYLUO1991/FaceParsing



import os
import sys
import cv2
import glob
import numpy as np

from PIL import Image

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 
              'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

def resize_and_move(ori_path, dest_path):
    dirs = os.listdir(ori_path)
    for item in dirs:
        print(item)
        if os.path.isfile(ori_path+item):
            im = Image.open(ori_path+item)
            imResize = im.resize((512,512), Image.ANTIALIAS)
            imResize.save(dest_path+item, 'JPEG', quality=90)

if __name__ == "__main__":
    root_path = '/home/yuhui/teamdrive/dataset/face_parse/CelebAMask-HQ/'
    val_folder = root_path + 'val/image/'
    resized_val_folder = root_path + 'val/image_resize/'
    make_folder(resized_val_folder)
    resize_and_move(val_folder, resized_val_folder)
