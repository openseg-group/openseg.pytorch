#!/usr/bin/python
# -*- encoding: utf-8 -*-
# Reference: https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py
#            

import os
import pdb
import shutil
import pandas as pd
from shutil import copyfile

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

if __name__ == "__main__":
    root_path = '/home/yuhui/teamdrive/dataset/face_parse/CelebAMask-HQ/'
    #### source data path
    s_label = root_path + 'CelebAMask-HQ-mask'
    s_img = root_path + 'CelebA-HQ-img'
    #### destination training data path
    d_train_label = root_path + 'train/label'
    d_train_img = root_path + 'train/image'
    #### destination testing data path
    d_test_label = root_path + 'test/label'
    d_test_img = root_path + 'test/image'
    #### val data path
    d_val_label = root_path + 'val/label'
    d_val_img = root_path + 'val/image'

    #### make folderYY
    make_folder(d_train_label)
    make_folder(d_train_img)
    make_folder(d_test_label)
    make_folder(d_test_img)
    make_folder(d_val_label)
    make_folder(d_val_img)

    #### calculate data counts in destination folder
    train_count = 0
    test_count = 0
    val_count = 0

    image_list = pd.read_csv(root_path + 'CelebA-HQ-to-CelebA-mapping.txt', delim_whitespace=True, header=None)
    # f_train = open('train_list.txt', 'w')
    # f_val = open('val_list.txt', 'w')
    # f_test = open('test_list.txt', 'w')

    for idx, x in enumerate(image_list.loc[:, 1]):
        print (idx, x)
        # if idx < 14700:
        #     continue
        # pdb.set_trace()
        if x >= 162771 and x < 182638:
            # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_val_label, str(val_count)+'.png'))
            # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_val_img, str(val_count)+'.jpg'))        
            val_count += 1
        elif x >= 182638:
            copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_test_label, str(test_count)+'.png'))
            copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_test_img, str(test_count)+'.jpg'))
            test_count += 1
        else:
            # copyfile(os.path.join(s_label, str(idx)+'.png'), os.path.join(d_train_label, str(train_count)+'.png'))
            # copyfile(os.path.join(s_img, str(idx)+'.jpg'), os.path.join(d_train_img, str(train_count)+'.jpg'))
            train_count += 1

    print (train_count + test_count + val_count)
    #### close the file
    # f_train.close()
    # f_val.close()
    # f_test.close()
