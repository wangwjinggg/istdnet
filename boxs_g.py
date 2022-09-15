from __future__ import division
import imghdr
from tkinter import W, image_names
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
from skimage.color import rgb2gray
import random
import numpy
from scipy import misc
import matplotlib
import math
import argparse
import pandas as pd

from torch import float64

#确定中心位置
def get_multi_center(img):

    H, W = img.shape
    #CX=[]
    #CY=[]
    boxes = []

    # convert the grayscale image to binary image
    res, thresh = cv2.threshold(img, 0, 255, 0) #将像素值大于127的统一赋值255（白）小于127的统一赋值为0（黑）。https://blog.csdn.net/a19990412/article/details/81172426

    # find contours in the binary image
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:#image,
        
        x, y, w, h = cv2.boundingRect(c)
        x1 = x-1
        x2 = x+w 
        y1 = y-1
        y2 = y+h 

        #cX = float(x)+ (w-1)/2
        #cY = float(y)+ (h-1)/2
             #M = cv2.moments(c)
             #cX = float(M["m10"]/(M["m00"]))#+0.1+0.00000001
             #cY = float(M["m01"]/(M["m00"]))#+0.1+0.00000001
        
        #x1 = int(cX+0.5)-15 #, int(cY-15+0.5), int(cX+15+0.5), int(cY+15+0.5)
        #y1 = int(cY+0.5)-15
        #x2 = int(cX+0.5)+15
        #y2 = int(cY+0.5)+15
        if x1<0:
            x1 = 0
            #x2 = 30
        if y1<0:
            y1 = 0
            #y2 = 30
        if x2>W-1:
            x2 = W-1
            #x1 = W-1-30
        if y2>H-1:
            y2 = H-1
            #y1 = H-1-30
        boxes.append([x1, y1, x2, y2]) #W-1, H-1是因为W=128时实际像素值范围是(0~127)，并没有128.

    return boxes, w, h #CX,CY,


if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default='/home/yons/data3/wwj/data/cGAN_data/training1/', ###/home/yons/data3/wwj/data/sirst-master/labels/###/home/yons/data3/wwj/data/IRSTD-1k/IRSTD1k_Label/#/home/yons/data3/wwj/erro_cGan_d
                        help="path of training ground truth")#/home/yons/data3/wwj/data/sirst-master/labels/
    parser.add_argument("--save_csv_path", type=str, default='/home/yons/data3/wwj/iaanet/cGAN_data/train_1_box_gt.csv', help="path to save box ground truth")
    parser.add_argument("--save_img_path", type=str, default="/home/yons/data3/wwj/cGAN_data_labels_0/", help="path to save results")
    parser.add_argument("--bord", type=int, default=8, help="在原小目标 expand mask's edge to generate box")
    args = parser.parse_args()

    bord = args.bord
    path = args.path
    save_csv_path = args.save_csv_path
    save_img_path =args.save_img_path

    images = os.listdir(args.path)
    message = []
    max_w, max_h = 0, 0
    for image in images:
        if image.endswith('_1.png'):
            continue
        img = cv2.imread(os.path.join(path, image), 0)

        Boxes, w, h = get_multi_center(img)
        
        Boxes1 = [' '.join(str(int(i)) for i in item) for item in Boxes]
        BoxesString = ";".join(Boxes1)

        message.append([image.replace('_2','_1'), BoxesString])#
        #img1 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        #for i in range(len(Boxes)):
        #        a=Boxes#[i]
                #for j in range(len(a)):             
        #        ret = cv2.rectangle(img1,(a[i][0], a[i][1]), (a[i][2], a[i][3]), (0, 255, 0), 1)
        #cv2.imwrite(os.path.join(save_img_path, image.replace('jpg','png')), ret)
        print(f'{image} record')
        if w>max_w:
            max_w = w 
        if h>max_h:
            max_h = h

    print(f'最大高{max_h},最大宽{max_w}')
    message = pd.DataFrame(message, columns=['image_name', 'BoxesString'])
    #save bounding boxes message into a csv
    message.to_csv(save_csv_path, index=False)
