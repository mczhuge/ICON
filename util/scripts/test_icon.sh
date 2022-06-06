echo "Model: ICON"
echo "Title: Salient Object Detection via Integrity Learning"
echo "Paper: https://arxiv.org/pdf/2101.07663.pdf"
echo "Code: https://github.com/mczhuge/ICON"
echo "=== Test ICON ==="


echo "Now Testing ICON-S..."
python main/test.py \
    --model 'ICON-S' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-S/ICON-Swin.weight' 

echo "Now Testing ICON-M..."
python main/test.py \
    --model 'ICON-M' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-M/ICON-CycleMLP.weight' 

echo "Now Testing ICON-P..."
python main/test.py \
    --model 'ICON-P' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-P/ICON-PVT.weight' 

echo "Now Testing ICON-R..."
python main/test.py \
    --model 'ICON-R' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-R/ICON-Res.weight' 

echo "Now Testing ICON-V..."
python main/test.py \
    --model 'ICON-V' \
    --task 'SOD' \
    --ckpt 'checkpoint/ICON/ICON-V/ICON-VGG.weight' 

echo "Now Testing ICON-R on SOC..."
python main/test.py \
    --model 'ICON-R' \
    --task 'SOC' \
    --ckpt 'checkpoint/ICON/ICON-R/ICON-Res-SOC.weight' 

# test SOC (Trained on SOC)
echo "Now Testing ICON-R on DUTS..."
python main/test.py \
    --model 'ICON-R' \
    --task 'SOC' \
    --ckpt 'checkpoint/ICON/ICON-R/ICON-Res.weight' 

