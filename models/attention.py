import torch
from torch import nn
from torch._C import dtype
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy import io 
import cv2
from utils.general import*
from .embedding import*
from .backbone import *

#filter out extremely tiny bounding boxes
def tiny_filter(boxes):
    """
    Input:
        boxes: x1, y1, x2, y2, conf
    """
    delta_w = boxes[:, 2] - boxes[:, 0]
    delta_h = boxes[:, 3] - boxes[:, 1]

    keep = (delta_w >=1) & (delta_h >=1)

    return boxes[keep]

def RandomSelectNegRegion(x, num, smin=5, smax=8):
    C, H, W = x.shape
    #random select number of negative boxes
    #num = random.randint(1, num)
    num = num
    neg_boxes = []
    for n in range(num):
        cx = random.randint(smax, W-smax)
        cy = random.randint(smax, H-smax)
        rw = random.randint(smin, smax)
        rh = random.randint(smin, smax)

        neg_boxes.append(torch.tensor([cx-rw, cy-rh, cx+rw, cy+rh, 0.5], dtype=torch.float))
    if num == 0:
        neg_boxes = None
    else:
        neg_boxes = torch.stack(neg_boxes, dim=0)
    return neg_boxes


class attention(nn.Module):
    def __init__(self, transformer, region_module, pos='cosin', d_model=256):
        super().__init__()
        self.transformer = transformer
        self.region_module = region_module
        self.seman_module = UNet(d_model=d_model)#semantic_estab(d_model=d_model)#resnet1()
        self.pos = pos
        
        if pos == 'cosin':
            self.posembedding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
        elif pos == 'learned':
            self.posembedding = PositionEmbeddingLearned(num_pos_feats=d_model // 2, num_embedding=700)

        self.pro_embed = MLP(d_model, d_model, 1, 3)
        self.d_model = d_model

    def forward(self, x, max_det_num=5, conf_thres=0.2, iou_thres=0.4, scale=20, expand=10, topk=5, fast=False):#True
        b, c, h, w = x.shape
        #get semantic feature map of the whole input images
        #if (self.training) | (not fast):
        seman_feat = self.seman_module(x) #seman_feat：SG模块的输出(8,512,128,128) 

        #position embedding the whole image
        if self.pos is None:
            image_poses = torch.zeros((b, self.d_model, h, w), device=x.device)
        else:
            image_poses = self.posembedding(x) #加上位置编码
        
        
        detect_output, region_boxes = self.region_module(x) #RPN模块
        region_boxes = region_boxes.detach()
        
        #For self-attn
        max_words_num = (scale**2)*max_det_num 
        region_words = []
        region_poses = []
        mask_maps = torch.ones((b, max_words_num), device=x.device, dtype=torch.bool)#8,128,128mask area out of proposed boxes

        target_boxes = []
        mask_boxes = []
        

        region_boxes_exists = True #whether detector find targets (Inference)
        
        for i in range(b):   #每个batch循环
            r_boxes = region_boxes[i]
            image_pos = image_poses[i]
            #NMS#非极大抑制
            #boxes = non_max_suppression(r_boxes, conf_thres=conf_thres, iou_thres=iou_thres) 
            
            if (r_boxes[:,4] > conf_thres).sum() > 0:
                boxes = non_max_suppression(r_boxes, conf_thres=conf_thres, iou_thres=iou_thres)
                #filter out boxes that are too small
                boxes = tiny_filter(boxes)
                boxes = boxes[:topk]

            else:
                boxes = non_max_suppression(r_boxes, conf_thres=0.0, iou_thres=iou_thres)
                #filter out boxes that are too small
                boxes = tiny_filter(boxes)
                boxes = boxes[:topk]      
            #现在的逻辑是，有>conf_thres，就正常去通过nms算框，没有的话降低阈值算完框取前topk个。
            
            #If detector proposed no region boxes, record and jump out (Inference).
            if (self.training==False) & (len(boxes)==0):
                region_boxes_exists = False
                break

            #if len(boxes) < max_det_num:
                #Select negative region
            #    neg_boxes = RandomSelectNegRegion(x[i], max_det_num-len(boxes), len(boxes))#选负样本
                #combine
            #    if neg_boxes is not None:
            #        neg_boxes = neg_boxes.to(boxes.device)
            #        boxes = torch.cat([boxes, neg_boxes], dim=0)
                #Keep no overlap
            #    boxes = non_max_suppression(boxes, conf_thres=0.0, iou_thres=iou_thres)

            elif self.training:
                boxes = boxes[:max_det_num]

            boxes = boxes[:,:4]

            target_box = []
            mask_box =[]

            region_word = torch.ones((h, w, self.d_model), device=x.device)
            region_pos = torch.ones((h, w, self.d_model), device=x.device)
            mask_map = mask_maps[i]
            words = []
            poses = []
            mask_word = torch.zeros((scale**2, self.d_model),device=x.device)
            mask_pos = torch.zeros((scale**2, self.d_model),device=x.device)

            for box in boxes:
                x11, y11, x21, y21 = (box[0]+0.5).to(torch.int), (box[1]+0.5).to(torch.int), (box[2]+0.5).to(torch.int), (box[3]+0.5).to(torch.int)#取整
                target_box.append([x11.item(), y11.item(), x21.item(), y21.item()])
                
                cX = (x11+x21)/2
                cY = (y11+y21)/2
                x1, y1, x2, y2 = int(cX+0.5)-int(scale/2), int(cY+0.5)-int(scale/2), int(cX+0.5)+int(scale/2), int(cY+0.5)+int(scale/2)
                if x1<0:
                    x1 = 0
                    x2 = scale
                if y1<0:
                    y1 = 0
                    y2 = scale
                if x2>w-1:
                    x2 = w-1
                    x1 = w-1-scale
                if y2>h-1:
                    y2 = h-1
                    y1 = h-1-scale
                mask_box.append([x1, y1, x2, y2]) #.item().item().item().item()      ###需要把取得框的位置传后面去。重叠的地方取更大的值？
                #if (self.training) | (not fast):
                    #get region's semantic feature
                #    r_seman_feat = seman_feat[i, :, y1:y2, x1:x2]
                #else:                                                                  ####测试时候是怎么搞的！
                #    cw, ch = (x2-x1), (y2-y1)        
                    #get region's semantic feature
                    #get local semantic of input images
                #    seman_feat = self.seman_module(x[i,:,max(0,y1-expand):min(h,y2+expand), max(0,x1-expand):min(w,x2+expand)].unsqueeze(dim=0))
                    ##这里
                #    ly1 = y1 if y1-expand < 0 else expand
                #    ly2 = ly1 + ch
    
                #    lx1 = x1 if x1-expand < 0 else expand
                #    lx2 = lx1 + cw
                    
                #    seman_feat = seman_feat.squeeze(dim=0)
                #    r_seman_feat = seman_feat[:, ly1:ly2, lx1:lx2]

                r_seman_feat = seman_feat[i, :, y1:y2, x1:x2]
                r_seman_feat = r_seman_feat.permute(1,2,0).contiguous() #9，10，512#框对应的特征图#C Y X ->YXC  30 30 256
                word = r_seman_feat
                word = word.view(-1, self.d_model)   #400 256
                words.append(word)
                pos = image_pos[:, y1:y2, x1:x2].permute(1, 2, 0).contiguous()
                pos = pos.view(-1, self.d_model)
                poses.append(pos)
                #region_word[y1:y2, x1:x2, :] = word #
                #region_pos[y1:y2, x1:x2, :] = image_pos[:, y1:y2, x1:x2].permute(1, 2, 0)
                mask_map[:(scale**2)*len(boxes)] = False

            if len(boxes) < max_det_num:
                for i in range(0, max_det_num-len(boxes)) :               
                    words.append(mask_word)                       ###疑问，这里的words生成的是list，0 1 2这种。或许应该搞成.cat？直接就是tensor？
                    poses.append(mask_pos)

            target_boxes.append(target_box)
            mask_boxes.append(mask_box)

            region_word = words[0] #region_word[~mask_map]#478，512
            for i in range(0, len(words)-1):
                region_word = torch.cat([region_word, words[i+1]], dim=0)
            region_pos = poses[0] #region_pos[~mask_map]
            for i in range(0, len(words)-1):
                region_pos = torch.cat([region_pos, poses[i+1]], dim=0)

            region_words.append(region_word)
            region_poses.append(region_pos)

            #max_words_num = (scale**2)*max_det_num                       ###这个数量应该是可以直接算出来的
            #if len(region_word) > max_words_num:#region_
            #    max_words_num = len(region_word) #region_                  ###这里只是保证每个batch的长度是一样的。

        if (self.training==False) & (region_boxes_exists==False):
            seg_output = None
            target_box = None
            region_boxes = None
        else:
            #pad words/poses length of each batch to be equal
            region_mask = torch.ones((b, max_words_num), device=region_word.device, dtype=torch.bool) #生成一个全T的 #8 ，591
            region_words_pad = torch.zeros((b, max_words_num, self.d_model), device=region_word.device, dtype=region_word.dtype)
            region_poses_pad = torch.zeros((b, max_words_num, self.d_model), device=region_word.device, dtype=region_word.dtype)
            for i, w in enumerate(region_words):  #region_words#i=8 w=每个图的特征 n*512， n的最大值为max_words_num
                
                #l, _ = w.shape  #每个图 n不一样，这里把n取出来，为l的值
                region_words_pad[i,:,:] = w #l
            #for i, h in enumerate(poses):
                region_poses_pad[i,:,:] = region_poses[i]#region_ l
            #for i in range(0, len(boxes)):
                region_mask[i,:] = mask_maps[i,:]
                                     #8 591 512             8 591 512    布尔，有值的地方为F，0的地方为T
                #w = w.sigmoid().detach().cpu().numpy()
                #w = np.uint8(w* 255)
                #io.savemat('w1.mat',{'words':w})
                #cv2.imwrite(os.path.join('/home/yons/data3/wwj/cGAN_data_labels_0/1.png'), w)                    
            output = self.transformer(region_words_pad, region_poses_pad, region_mask)#自注意力计算
            output = output.permute(1, 0, 2)

            seg_output = self.pro_embed(output) #MLP
 

        return (detect_output, seg_output, mask_maps, mask_boxes) if self.training else (seg_output, mask_maps, region_boxes, mask_boxes)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
