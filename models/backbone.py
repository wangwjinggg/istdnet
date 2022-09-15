import torch
import torch.nn.functional as F
from torch import nn
import torchvision

from utils.general import*
from .unet_parts import *

class backbone(nn.Module): #RPN的backbone
    def __init__(self):
        super().__init__()
        backbone = torchvision.models.resnet.resnet18(pretrained=True) #使用的是在ImageNet上预训练过的模型
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False)

        return_layer = {"layer3": "0"} #对应下面，这里应该指的是抽取layer3层输出对应[0]。
        self.stride = 16   #这个是指啥
        self.num_channel = 256
        self.anchor = torch.tensor([8, 8])

        #抽取网络中间层输出：
        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layer)
        #x y w h obj 
        self.detect = nn.Conv2d(self.num_channel, 5, kernel_size=(1,1), stride=(1,1))

    def forward(self, x, device='cuda'):
        src = self.body(x)
        src = src['0']
        x = self.detect(src) #卷积后的输出
        
        
        bs, _, ny, nx = x.shape
        x = x.permute(0, 2, 3, 1)
        
        #if not self.training:
        #get bounding box 
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])  #torch.meshgrid：将图像划分为单元网格，torch.arange(ny)：生成1-（ny-1）的int型一维张量
        grid = torch.stack((xv, yv), 2).view((1, ny, nx, 2)).float().to(device) #torch.stack：沿一个新的维度对输入张量序列进行拼接，序列中张量应该为同一形状

        y = x.sigmoid()
        xy = (y[..., 0:2] * 2 - 0.5 + grid) * self.stride
        wh = (y[..., 2:4] * 2) ** 2 * self.anchor.view(1, 1, 1, 2).to(device)
        #wh = (y[..., 2:4] * 0 + 1)* self.anchor.view(1, 1, 1, 2).to(device) #self.anchor.view(8, 8, 8, 2).to(device)# ** 2 
        y = torch.cat((xy, wh, y[..., 4:]), -1)

        return x, y.view(bs, -1, 5)

class region_propose(nn.Module): #RPN模块
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    
    def forward(self, x):
        _, _, h, w = x.shape
        detect_output, boxes = self.backbone(x) #backbone见上
        boxes[..., :4] = xywh2xyxy(boxes[..., :4])
        #clamp
        boxes[...,[0,2]]= boxes[...,[0,2]].clamp(0, w)#把预测的box的值规定到图片内
        boxes[...,[1,3]]= boxes[...,[1,3]].clamp(0, h)
        #boxes[...,[0,1,2,3]] = (boxes[...,[0]]+0.5).to(torch.int), (boxes[...,[1]]+0.5).to(torch.int), (boxes[...,[2]]+0.5).to(torch.int), (boxes[...,[3]]+0.5).to(torch.int)#取整
        #if boxes[...,[0]] <0 :
        #    boxes[...,[0]],boxes[...,[2]] = 0 ,30   #这里需要换成可以修改的参数。
        #if boxes[...,[1]] <0 :
        #    boxes[...,[1]],boxes[...,[3]] = 0 ,30
        #if boxes[...,[2]] >w-1 :
        #    boxes[...,[0]],boxes[...,[2]] = w-1-30 ,w-1
        #if boxes[...,[3]] >h-1 :    
        #    boxes[...,[1]],boxes[...,[3]] = h-1-30 ,h-1    #这个在这里要计算64次取整操作，怎么样优化？
        
        return detect_output, boxes

#SG
class semantic_estab(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(512, d_model, kernel_size=(1,1), stride=(1,1), bias=False),
        )

    def forward(self, x):
        x = self.block1(x)
        layer1 = self.layer1(x)

        x = self.block2(layer1)
        
        return x 

class UNet(nn.Module):
    def __init__(self, d_model,n_channels=1, n_classes=2, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, d_model, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #x = self.up1(x5, x4)
        x = self.up2(x4, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = x#self.outc(x)
        return logits

class resnet1(nn.Module):
    def __init__(self):
        super().__init__()
        #nn.Conv2d(1, 64, kernel_size=(7,7), stride=(2,2), padding=(3,3), bias=False),
        backbone = torchvision.models.resnet.resnet18(pretrained=True) #使用的是在ImageNet上预训练过的模型
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7,7), stride=(1,1), padding=(3,3), bias=False)
        backbone.maxpool = nn.MaxPool2d(kernel_size=1, stride=1, padding=0, dilation=1, ceil_mode=False)

        backbone.layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
  
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))

        backbone.layer3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
  
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True))


        return_layer = {"layer3": "0"} #对应下面，这里应该指的是抽取layer3层输出对应[0]。
        
        #抽取网络中间层输出：
        self.body = torchvision.models._utils.IntermediateLayerGetter(backbone, return_layers=return_layer)

    def forward(self, x):
        x = self.body(x)
        x = x['0']
        return x


if __name__ == '__main__':

    input = torch.rand(1,1,128,128)
    #backbone = backbone()#mode='bbox'
    model1 = resnet1()
    #model = region_propose(backbone)
    #model = model.to('cuda')
    model1 = model1.to('cuda')
    #output = model(input.to('cuda'))
    output1 = model1(input.to('cuda'))

    print(output1.shape)


