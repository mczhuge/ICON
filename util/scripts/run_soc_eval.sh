echo "Model: ICON"
echo "Title: Salient Object Detection via Integrity Learning"
echo "Paper: https://arxiv.org/pdf/2101.07663.pdf"
echo "Code: https://github.com/mczhuge/ICON"
echo "=== Test ICON on SOC (8 attributes) ==="

method_=ICON-R

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-AC' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-BO' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-CL' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-HO' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-MB' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-OC' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-OV' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-SC' \

python util/evaltool/soc_eval.py   \
    --method  $method_ \
    --dataset  'SOC' \
    --attr  'SOC-SO' \
