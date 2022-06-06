"""
Code: https://github.com/mczhuge/ICON
Author: mczhuge
Desc: Core code for evaluationg SOD
"""

import os
import sys
import cv2
from tqdm import tqdm
import metrics as M
import json
import argparse

def main():
    args = parser.parse_args()

    FM = M.Fmeasure_and_FNR()
    WFM = M.WeightedFmeasure()
    SM = M.Smeasure()
    EM = M.Emeasure()
    MAE = M.MAE()

    method = args.method
    dataset = args.dataset
    #attr = args.attr

    gt_root = os.path.join('datasets/'+dataset+'/Test/', 'GT')
    pred_root = os.path.join('util/evaltool/Prediction/'+method+'/', dataset)

    gt_name_list = sorted(os.listdir(pred_root))


    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        #gt_name_pre = gt_name.split('_sal_fuse')[0] #C2S
        #gt_name_pre = gt_name.split('_ras')[0]
        #gt_path = os.path.join(gt_root, gt_name.split('.')[0]+'.png')     ##Condinst    
        gt_path = os.path.join(gt_root, gt_name)     ##Condinst    
        #gt_path = os.path.join(gt_root, gt_name_pre+'.png') #c2s
        #gt_path = os.path.join(gt_root, gt_name_pre+'.png') #ras
        #print(gt_path)
        #CSF-R2
        #pred_name = gt_name_pre+'_sal_fuse.png' #C2S
        #pred_name = gt_name_pre+'_ras.png' #C2S
        pred_path = os.path.join(pred_root, gt_name)    
        #print(gt_path)
        #pred_path = os.path.join(pred_root, gt_name.split('.')[0]+'.jpg')   #Condinst
        #pred_path = os.path.join(pred_root, gt_name)
        #print(pred_path) 
        #pred_path = os.path.join(pred_root, pred_name) #C2S
        #pred_path = os.path.join(pred_root, pred_name) #ras
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        gt_width, gt_height = gt.shape
        pred_width, pred_height = pred.shape

        #print(gt.shape, pred.shape)
        if gt.shape != pred.shape:
            cv2.imwrite( os.path.join('util/evaltool/Prediction/'+method+'/'+dataset+'/', gt_name), cv2.resize(pred, gt.shape[::-1]))
        #print('OK')
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        FM.step(pred=pred, gt=gt)
        WFM.step(pred=pred, gt=gt)
        SM.step(pred=pred, gt=gt)
        EM.step(pred=pred, gt=gt)
        MAE.step(pred=pred, gt=gt)
        #FNR.step(pred=pred, gt=gt)
    
    fm = FM.get_results()[0]['fm']
    wfm = WFM.get_results()['wfm']
    sm = SM.get_results()['sm']
    em = EM.get_results()['em']
    mae = MAE.get_results()['mae']
    fnr = FM.get_results()[1]

    print(
        'Method:', args.method, ',',
        'Dataset:', args.dataset, '||',
        #'Attribute:', args.attr, '||',
        'Smeasure:', sm.round(3), '; ',
        'meanEm:', '-' if em['curve'] is None else em['curve'].mean().round(3), '; ',
        'wFmeasure:', wfm.round(3), '; ',
        'MAE:', mae.round(3), '; ',
        'adpEm:', em['adp'].round(3), '; ',
        'maxEm:', '-' if em['curve'] is None else em['curve'].max().round(3), '; ',
        'adpFm:', fm['adp'].round(3), '; ',
        'meanFm:', fm['curve'].mean().round(3), '; ',
        'maxFm:', fm['curve'].max().round(3), '; ',
        'fnr:', fnr.round(3),
        sep=''
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='ICON')
    parser.add_argument("--dataset", default='DUTS')
    #parser.add_argument("--attr", default='SOC-AC')
    main()
