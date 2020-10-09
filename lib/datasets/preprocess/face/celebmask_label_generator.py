#!/usr/bin/python
# -*- encoding: utf-8 -*-
# Reference: https://github.com/switchablenorms/CelebAMask-HQ/blob/master/face_parsing/Data_preprocessing/g_mask.py
#            

# other resource: 
#                   https://github.com/switchablenorms/CelebAMask-HQ
#                   https://github.com/zllrunning/face-parsing.PyTorch
#                   https://github.com/JACKYLUO1991/FaceParsing



import os
import cv2
import glob
import numpy as np

label_list = ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 
              'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))

if __name__ == "__main__":
    root_path = '/home/yuhui/teamdrive/dataset/face_parse/CelebAMask-HQ/'
    folder_base = root_path + 'CelebAMask-HQ-mask-anno'
    folder_save = root_path + 'CelebAMask-HQ-mask'
    img_num = 30000
    make_folder(folder_save)

    for k in range(14700, img_num):
        folder_num = k // 2000
        im_base = np.zeros((512, 512))
        for idx, label in enumerate(label_list):
            filename = os.path.join(folder_base, str(folder_num), str(k).rjust(5, '0') + '_' + label + '.png')
            if (os.path.exists(filename)):
                print (label, idx+1)
                im = cv2.imread(filename)
                im = im[:, :, 0]
                im_base[im != 0] = (idx + 1)

        filename_save = os.path.join(folder_save, str(k) + '.png')
        print (filename_save)
        cv2.imwrite(filename_save, im_base)


'''
# based on https://raw.githubusercontent.com/zllrunning/face-parsing.PyTorch/master/prepropess_data.py
import os.path as osp
import os
import cv2
from PIL import Image
import numpy as np
root_path = '/home/yuhui/teamdrive/dataset/face_parse/CelebAMask-HQ/'
face_data = root_path + 'CelebA-HQ-img'
face_sep_mask = root_path + 'CelebAMask-HQ-mask-anno'
mask_path = root_path + 'CelebAMaskHQ-mask'
counter = 0
total = 0
for i in range(15):

    atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
            'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']

    for j in range(i * 2000, (i + 1) * 2000):

        mask = np.zeros((512, 512))

        for l, att in enumerate(atts, 1):
            total += 1
            file_name = ''.join([str(j).rjust(5, '0'), '_', att, '.png'])
            path = osp.join(face_sep_mask, str(i), file_name)

            if os.path.exists(path):
                counter += 1
                sep_mask = np.array(Image.open(path).convert('P'))
                # print(np.unique(sep_mask))
                mask[sep_mask == 225] = l
        cv2.imwrite('{}/{}.png'.format(mask_path, j), mask)
        print(j)
print(counter, total)
'''

