#!/usr/bin/python3
#coding=utf-8

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models
#from torch.utils import model_zoo
#from torchvision.models.resnet import resnet50

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        feats = list(models.vgg16_bn(pretrained=True).features.children())
        self.conv1 = nn.Sequential(*feats[:6])
        self.conv2 = nn.Sequential(*feats[6:13])
        self.conv3 = nn.Sequential(*feats[13:23])
        self.conv4 = nn.Sequential(*feats[23:33])
        self.conv5 = nn.Sequential(*feats[33:43])

        self.initialize()

        for m in self.modules():
            if isinstance(m, nn.ReLU) or isinstance(m, nn.Dropout):
                m.inplace = True
         


    def forward(self, x):
        input = x

        E1 = self.conv1(x)
        E2 = self.conv2(E1)
        E3 = self.conv3(E2)
        E4 = self.conv4(E3)
        E5 = self.conv5(E4)
        #print(E5.size())

        return E5, E4, E3, E2, E1#E1, E2, E3, E4, E5

    def initialize(self):
        pass
        #weight_init(self)



if __name__ == "__main__":
    model = VGG()
    input = torch.autograd.Variable(torch.randn(4, 3, 384, 384))
    output = model(input)
    print(output[1].size())


