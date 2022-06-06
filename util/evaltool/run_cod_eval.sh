echo "Code: https://github.com/mczhuge/ICON"
echo "Author: mczhuge"
echo "Desc: Eval Camouflaged Object Detection (COD)"


method_=ICON-R

python sod_eval.py   \
    --method  $method_ \
    --dataset  'CAMO' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'COD10K' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'CHAMELEON' 

python sod_eval.py   \
    --method  $method_ \
    --dataset  'CPD1K' 

