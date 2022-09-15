from functools import total_ordering
from numpy.lib.function_base import average
import torch
import os
import numpy as np
import cv2
import argparse

#Metric F1: https://github.com/wanghuanphd/MDvsFA_cGAN/blob/master/demo_MDvsFA_pytorch.py
def calculateF1Measure(output_image,gt_image,thre): #阈值为0.5
    output_image = np.squeeze(output_image)
    gt_image = np.squeeze(gt_image)
    out_bin = output_image>thre
    gt_bin = gt_image>thre
    right_bin = np.sum(gt_bin*out_bin)
    total_bin = np.sum(gt_bin) + np.sum(out_bin)
    iou = np.sum(gt_bin*out_bin)/(total_bin-np.sum(gt_bin*out_bin))
    recall = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(gt_bin))
    prec   = np.sum(gt_bin*out_bin)/np.maximum(1,np.sum(out_bin))
    F1 = 2*recall*prec/np.maximum(0.001,recall+prec)
    return iou, prec, recall, F1, right_bin, total_bin 

def test1(image_path, mask_path, model, device, conf_thres, iou_thres, expand, topk, fast, scale=20):
    '''
    only support batch size = 1
    '''  
    average_F1 = 0
    average_prec = 0
    average_recall = 0
    niou = 0
    t_right_bin = 0
    t_total_bin = 0
    lengh = scale**2

    with torch.no_grad():
        num = len(os.listdir(image_path))
        for i, img_name in enumerate(os.listdir(image_path)):
            print(f'{i+1}/{num}', end='\r', flush=True)
            input_img = cv2.imread(os.path.join(image_path, img_name), 0) / 255.0
            target = cv2.imread(os.path.join(mask_path, img_name), 0) / 255.0
            input = torch.from_numpy(input_img).to(torch.float)
            input = input[None, None, :]
            _, _, h, w = input.shape
            
            output, mask_maps, _, mask_boxes = model(input.to(device), max_det_num=3, conf_thres=conf_thres, iou_thres=iou_thres, expand=expand, topk=topk, fast=fast)

            probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
            if output is not None:
                output = output.squeeze()
                output = output.sigmoid()
                mask_maps = mask_maps.squeeze()

                for k ,w in enumerate(mask_boxes):
                    mask_box = w #mask_boxes[i]
                    for j ,h in enumerate(mask_box):

                        probability_map[h[1]:h[3],h[0]:h[2]] = output[(j*lengh) : ((j+1)*lengh)].view(scale,scale)
                        ##统一的框的宽度需要传过来
                #probability_map[~mask_maps] = output

            probability_map = probability_map.cpu().numpy()
            iou, prec, recall, F1, right_bin, total_bin  = calculateF1Measure(probability_map, target, 0.5)
            niou = (niou * i + iou) / (i + 1)
            average_F1 = (average_F1 * i + F1) / (i + 1)
            average_prec = (average_prec * i + prec) / (i + 1)
            average_recall = (average_recall * i + recall) / (i + 1)
            t_right_bin += right_bin                
            t_total_bin += total_bin 
        Iou = t_right_bin /(t_total_bin - t_right_bin)

    print(f'niou:{niou}  iou:{Iou}   prec:{average_prec}  recall:{average_recall}  F1: {average_F1} ')

    return average_F1

if __name__ == '__main__':
    #######################################
    #set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_image", type=str, default="/home/yons/data3/wwj/data/sirst-master/images/", help="path to load testing image")#/home/yons/data3/wwj/data/IRSTD-1k/IRSTD1k_Img/
    parser.add_argument("--test_mask", type=str, default="/home/yons/data3/wwj/data/sirst-master/labels/", help="path to load testing masks")#/home/yons/data3/wwj/data/IRSTD-1k/IRSTD1k_Label/
    parser.add_argument('--weights', type=str, default="/home/yons/data3/wwj/iaanet/outputs/demo-bb1-10e/best.pt", help="path of the weights")
    
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.7, help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=3, help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--expand", type=int, default=8, help="The additonal side length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', help='fast inference')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####################################
    model = torch.load(args.weights)
    model.to(device)
    model.eval()
    
    test1(args.test_image, args.test_mask, model, device, args.conf_thres, args.iou_thres, args.expand, args.topk, args.fast)
    