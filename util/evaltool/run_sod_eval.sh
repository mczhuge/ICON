echo "Code: https://github.com/mczhuge/ICON"
echo "Author: mczhuge"
echo "Desc: Eval SOD (6 common benchmarks)"


method_=ICON-V


python sod_eval.py   \
    --method  $method_ \
    --dataset  'ECSSD' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'PASCAL-S' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'DUTS' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'HKU-IS' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'DUT-OMRON' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'SOD' 


