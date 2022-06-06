#!/usr/bin/python3
#coding=utf-8

import torch
from torch import nn
from torch.utils import model_zoo
from torchvision.models.resnet import resnet50
from .swin_encoder import SwinTransformer
from .resnet_encoder import ResNet
from .vgg_encoder import VGG
from .pvtv2_encoder import pvt_v2_b4
from .cyclemlp_encoder import CycleMLP_B4
from .modules import DFA, ICE, PWV
from timm.models import create_model
#from pytorch_grad_cam import GradCAM, ScoreCAM
#from pytorch_grad_cam.utils.image import show_cam_on_image
import collections
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def weight_init_backbone(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear): 
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            #nn.init.xavier_normal_(m.weight, gain=1)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, (nn.ReLU, nn.Sigmoid, nn.PReLU, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool1d, nn.Sigmoid, nn.Identity)):
            pass
        else:
            m.initialize()

class Classifier_Module(nn.Module):
    def __init__(self,dilation_series,padding_series,NoLabels, input_channel):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation,padding in zip(dilation_series,padding_series):
            self.conv2d_list.append(nn.Conv2d(input_channel,NoLabels,kernel_size=3,stride=1, padding =padding, dilation = dilation,bias = True))
        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list)-1):
            out += self.conv2d_list[i+1](x)
        return out

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(in_planes, out_planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_planes)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x

class ICON(torch.nn.Module):
    def __init__(self, cfg, model_name='ICON-R'):
        super(ICON, self).__init__()
        self.cfg = cfg
        self.model_name = model_name        

        if self.model_name == 'ICON-S':
            ### Swin Encoder ###
            self.encoder = SwinTransformer(img_size=384, 
                                           embed_dim=128,
                                           depths=[2,2,18,2],
                                           num_heads=[4,8,16,32],
                                           window_size=12)

            pretrained_dict = torch.load('checkpoint/Backbone/Swin/swin_base_patch4_window12_384_22k.pth')["model"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)

            self.dfa1 = DFA(1024, 64)
            self.dfa2 = DFA(512, 64)
            self.dfa3 = DFA(256, 64)
            self.dfa4 = DFA(128, 64)
       

        elif self.model_name == 'ICON-P':
            ### PVT Encoder ###
            self.encoder = pvt_v2_b4()

            pretrained_dict = torch.load('checkpoint/Backbone/PVTv2/pvt_v2_b4.pth')
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in self.encoder.state_dict()}
            self.encoder.load_state_dict(pretrained_dict)

            self.dfa1 = DFA(512, 64)
            self.dfa2 = DFA(320, 64)
            self.dfa3 = DFA(128, 64)
            self.dfa4 = DFA(64, 64)

        elif self.model_name == 'ICON-R':
            ### ResNet Encoder ###
            # loader_download_version       
            self.encoder = ResNet(self.cfg) 
            pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
            encoder_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            encoder_dict.update(pretrained_dict)
            self.encoder.load_state_dict(encoder_dict) 
            # Or load torch_model_version
            #self.encoder = ResNet() # torch_model_version 

            self.dfa1 = DFA(2048, 64)
            self.dfa2 = DFA(1024, 64)
            self.dfa3 = DFA(512, 64)
            self.dfa4 = DFA(256, 64)
        
        elif self.model_name == "ICON-V":
            #### VGG Encoder ####
            self.encoder = VGG()
            self.dfa1 = DFA(512, 64)
            self.dfa2 = DFA(512, 64)
            self.dfa3 = DFA(256, 64)
            self.dfa4 = DFA(128, 64)

        elif self.model_name == "ICON-M":
            #### MLP Encoder ####
            self.encoder = CycleMLP_B4()
 
            pretrained_dict = torch.load('checkpoint/Backbone/CycleMLP/CycleMLP_B4.pth')
            pretrained_dict = pretrained_dict['model']
            encoder_dict = self.encoder.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in encoder_dict}
            self.encoder.load_state_dict(pretrained_dict, strict=False)

            self.dfa1 = DFA(512, 64)
            self.dfa2 = DFA(320, 64)
            self.dfa3 = DFA(128, 64)
            self.dfa4 = DFA(64, 64)
      
        else:
            print("UNDEFINED BACKBONE NAME.")

        self.ice1 = ICE()
        self.ice2 = ICE()
        self.pwv = PWV(64)
        self.fuse  = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1), nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.predtrans1  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans2  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans3  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans4  = nn.Conv2d(64, 1, kernel_size=3, padding=1)
        self.predtrans5  = nn.Conv2d(64, 1, kernel_size=3, padding=1)

        self.initialize()

    def _make_pred_layer(self, block, dilation_series, padding_series, NoLabels, input_channel):
        return block(dilation_series, padding_series, NoLabels, input_channel)

    def forward(self, x, shape=None, name=None):

        features = self.encoder(x)

        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        if len(features) > 4:
            x5 = features[4]   
     
        if self.model_name == 'ICON-S':
            x1 = self.dfa1(x2)
            x2 = self.dfa2(x3)
            x3 = self.dfa3(x4)
            x4 = self.dfa4(x5)
        else:
            x1 = self.dfa1(x1) 
            x2 = self.dfa2(x2) 
            x3 = self.dfa3(x3) 
            x4 = self.dfa4(x4)   

        x1  = self.ice1(in1=x1, in2=x2)
        x2  = self.ice1(in1=x2, in2=x1, in3=x3)
        x3  = self.ice2(in1=x3, in2=x2, in3=x4)
        x4  = self.ice2(in1=x4, in2=x3)
        x1, x2, x3, pose = self.pwv(x1, x2, x3)

        if shape is None:
            shape = x.size()[2:]

        x3 = F.interpolate(x3,   size=x4.size()[2:], mode='bilinear')
        x4  = self.fuse(x4*x3) + x4

        pred1  = F.interpolate(self.predtrans1(x1),   size=shape, mode='bilinear')
        pred2  = F.interpolate(self.predtrans2(x2),   size=shape, mode='bilinear')       
        pred3  = F.interpolate(self.predtrans3(x3),   size=shape, mode='bilinear')
        pred4  = F.interpolate(self.predtrans4(x4),   size=shape, mode='bilinear')
        pose   = F.interpolate(self.predtrans5(pose), size=shape, mode='bilinear')

        return pred1, pred2, pred3, pred4, pose

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)

if __name__ == '__main__':
  
    import torch
    from ptflops import get_model_complexity_info

    with torch.cuda.device(0):
       net = ICON()
       #ICON-S and ICON-P: (3, 384, 384), Others: (3, 352, 352)
       macs, params = get_model_complexity_info(net, (3, 352, 352), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
       print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
       print('{:<30}  {:<8}'.format('Number of parameters: ', params))

       
    


