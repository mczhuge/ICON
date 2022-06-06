"""
Code: https://github.com/mczhuge/ICON
Author: mczhuge
Desc: Core code for validation
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
    #FNR = M.FNR()

    method = args.method
    dataset = args.dataset
    #attr = args.attr

    #SOD
    gt_root = os.path.join('datasets/'+dataset+'/Test/', 'GT')


    pred_root = os.path.join('util/evaltool/Prediction/'+method+'/', dataset)

    print(gt_root)

    gt_name_list = sorted(os.listdir(pred_root))

    for gt_name in tqdm(gt_name_list, total=len(gt_name_list)):
        #print(gt_name)
        gt_path = os.path.join(gt_root, gt_name)
        pred_path = os.path.join(pred_root, gt_name)
        gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        gt_width, gt_height = gt.shape
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        pred_width, pred_height = pred.shape
        #print(gt.shape, pred.shape)
        if gt.shape != pred.shape:
            cv2.imwrite( os.path.join('util/evaltool/Prediction/'+method+'/'+dataset+'/', gt_name), cv2.resize(pred, gt.shape[::-1]))
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

    Method_r = str(args.method)
    Dataset_r = str(args.dataset)
    Smeasure_r = str(sm.round(3))
    Wmeasure_r = str(wfm.round(3))
    MAE_r = str(mae.round(3))
    adpEm_r = str(em['adp'].round(3))
    meanEm_r = str('-' if em['curve'] is None else em['curve'].mean().round(3))
    maxEm_r = str('-' if em['curve'] is None else em['curve'].max().round(3))
    adpFm_r = str(fm['adp'].round(3))  
    meanFm_r = str(fm['curve'].mean().round(3))
    maxFm_r = str(fm['curve'].max().round(3))
    fnr_r = str(fnr.round(3))


    eval_record = str(        
        'Method:'+ Method_r + ','+
        'Dataset:'+ Dataset_r + '||'+
        'Smeasure:'+ Smeasure_r + '; '+
        'meanEm:'+ meanEm_r + '; '+
        'wFmeasure:'+ Wmeasure_r + '; '+
        'MAE:'+ MAE_r + '; '+
        'fnr:'+ fnr_r + '||' +
        'adpEm:'+ adpEm_r + '; '+
        'meanEm:'+ meanEm_r + '; '+
        'maxEm:'+ maxEm_r + '; '+
        'adpFm:'+ adpFm_r + '; '+
        'meanFm:'+ meanFm_r + '; '+
        'maxFm:'+ maxFm_r
         )


    print(eval_record)
    print('#'*50)

    txt = 'util/evaltool/Prediction/Train_record.txt'
    f = open(txt, 'a')
    f.write(eval_record)
    f.write("\n")
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default='ICON')
    parser.add_argument("--dataset", default='DUTS')
    #parser.add_argument("--attr", default='SOC-AC')
    main()
