echo "Model: ICON"
echo "Title: Salient Object Detection via Integrity Learning"
echo "Paper: https://arxiv.org/pdf/2101.07663.pdf"
echo "Code: https://github.com/mczhuge/ICON"
echo "=== Eval SOD (6 common benchmarks) ==="


method_=ICON-P

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'ECSSD' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'PASCAL-S' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'DUTS' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'HKU-IS' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
   --dataset  'DUT-OMRON' 

python util/evaltool/sod_eval.py   \
    --method  $method_ \
    --dataset  'SOD' 


