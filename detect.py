import torch
import os
import cv2
import numpy as np
import argparse


if __name__ == '__main__':
    #######################################
    #set up
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, default="/home/yons/data3/wwj/data/sirst-master/images/", help="path of the image folder/file")#/home/yons/data3/wwj/data/IRSTD-1k/IRSTD1k_Img
    parser.add_argument("--save_path", type=str, default="./inference/", help="path to save results")
    parser.add_argument('--folder', default=True, action='store_true', help=' or False  detect images in folder (default:image file)')
    parser.add_argument('--weights', type=str, default="/home/yons/data3/wwj/iaanet/outputs/demo-bb1-10e/best.pt", help="path of the weights")
    parser.add_argument("--iou_thres", type=float, default=0.6, help="iou threshold for detection stage")
    parser.add_argument("--conf_thres", type=float, default=0.4, help="confidence threshold for detection stage")
    parser.add_argument("--topk", type=int, default=3, help="if predict no boxes, select out k region boxes with top confidence")
    parser.add_argument("--expand", type=int, default=8, help="The additonal length of expanded local region for semantic generator")
    parser.add_argument('--fast', action='store_true', help='fast inference')
    args = parser.parse_args()
    
    os.makedirs(args.save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #####################################
    #dataset 
    if args.folder:
        datalist = os.listdir(args.image_path)
    else:
        datalist = [(args.image_path).split('/')[-1]]
    #model
    Model = torch.load(args.weights)
    Model.to(device)
    Model.eval()
    scale=20
    lengh = scale**2

    with torch.no_grad():
        for img_path in datalist:
            if args.folder:
                input = cv2.imread(os.path.join(args.image_path, img_path), 0)
            else:
                input = cv2.imread(args.image_path, 0)
                
            h, w = input.shape
            img = input[None,None,:]
            img = np.float32(img) / 255.0

            input = torch.from_numpy(img)
            #max_det_num=0 for inference
            output, mask_maps, region_boxes, mask_boxes = Model(input.to(device), max_det_num=3, conf_thres=args.conf_thres, iou_thres=args.iou_thres, expand=args.expand, topk=args.topk, fast=args.fast)
        
            #segmentation results
            probability_map = torch.zeros((h, w), dtype=torch.float, device=device)
            if output is not None:
                output = output.squeeze()
                output = output.sigmoid()
                mask_maps = mask_maps.squeeze()

                for i ,w in enumerate(mask_boxes):
                    mask_box = w #mask_boxes[i]
                    for j ,h in enumerate(mask_box):
                        probability_map[h[1]:h[3],h[0]:h[2]] = output[(j*lengh) : ((j+1)*lengh)].view(scale,scale)
                #probability_map[~mask_maps] = output
                
            probability_map = probability_map.cpu().numpy()
            probability_map = np.uint8(probability_map * 255)

            #for i in range(len(mask_boxes)):
            #    a=mask_boxes[i]
            #    for j in range(len(a)):
            #        cv2.rectangle(probability_map,(a[j][0], a[j][1]), (a[j][2], a[j][3]), (255, 0, 0), 1)
            cv2.imwrite(os.path.join(args.save_path, img_path.replace('jpg','png')), probability_map)

            print(f'record: {img_path}')
