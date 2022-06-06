#!/bin/bash
echo "Code: https://github.com/mczhuge/ICON"
echo "Eval during traning..."

method_=${1}



python util/evaltool/sod_valid_sod.py   \
    --method  $method_ \
    --dataset  'ECSSD' 

python util/evaltool/sod_valid_sod.py   \
    --method  $method_ \
    --dataset  'PASCAL-S' 


:<<!
python util/evaltool/sod_valid_soc.py   \
    --method  $method_ \
    --dataset  'SOC/SOC-AC' 

python util/evaltool/sod_valid_soc.py   \
    --method  $method_ \
    --dataset  'SOC/SOC-BO' 

python util/evaltool/sod_valid_soc.py   \
    --method  $method_ \
    --dataset  'SOC/SOC-CL' 
!
