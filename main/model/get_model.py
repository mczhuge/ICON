#!/usr/bin/python3
#coding=utf-8

from model.icon.icon import ICON

def get_model(cfg, model_name, img_name=None):

    if model_name and img_name:
        model = ICON(cfg, model_name, img_name).cuda()
    else: 
        model = ICON(cfg, model_name).cuda()

    print("Model based on {} have {:.4f}Mb paramerters in total".format(model_name, sum(x.numel()/1e6 for x in model.parameters())))

    return model.cuda()
