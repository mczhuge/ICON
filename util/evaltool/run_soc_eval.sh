echo "Code: https://github.com/mczhuge/ICON"
echo "Author: mczhuge"
echo "Desc: Eval SOC (8 attributes)"

method_=ICON-R

python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-AC' \

python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-BO' \


python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-CL' \


python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-HO' \

python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-MB' \

python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-OC' \


python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-OV' \

python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-SC' \

python soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-SO' \




