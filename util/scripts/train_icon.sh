echo "Model: ICON"
echo "Title: Salient Object Detection via Integrity Learning"
echo "Paper: https://arxiv.org/pdf/2101.07663.pdf"
echo "Code: https://github.com/mczhuge/ICON"
echo "=== Training ICON ==="


echo "Now Training ICON-S..."
python main/train.py \
    --model 'ICON-S' \
    --dataset 'datasets/DUTS/Train' \
    --lr 2e-3 \
    --decay 2e-4 \
    --momen 0.9 \
    --batchsize 6 \
    --loss 'CPR' \
    --savepath 'checkpoint/ICON/ICON-S/' \
    --valid True


:<<!
echo "Now Training ICON-M..."
python main/train.py \
    --model 'ICON-M' \
    --dataset 'datasets/DUTS/Train' \
    --lr 2e-4 \
    --decay 1e-4 \
    --momen 0.9 \
    --batchsize 8 \
    --loss 'CPR' \
    --savepath 'checkpoint/ICON/ICON-M/' \
    --valid True
!

:<<!
echo "Now Training ICON-P..."
python main/train.py \
    --model 'ICON-P' \
    --dataset 'datasets/DUTS/Train' \
    --lr 0.05 \
    --decay 1e-4 \
    --momen 0.9 \
    --batchsize 10 \
    --loss 'CPR' \
    --savepath 'checkpoint/ICON/ICON-P/' \
    --valid True
!


:<<!
echo "Now Training ICON-R..."
python main/train.py \
    --model 'ICON-R' \
    --dataset 'datasets/DUTS/Train' \
    --lr 0.05 \
    --decay 1e-4 \
    --momen 0.9 \
    --batchsize 32 \
    --loss 'CPR' \
    --savepath 'checkpoint/ICON/ICON-R/' \
    --valid True
!


:<<!
echo "Now Training ICON-V..."
python main/train.py \
    --model 'ICON-V' \
    --dataset 'datasets/DUTS/Train' \
    --lr 0.05 \
    --decay 2e-5 \
    --momen 0.9 \
    --batchsize 8 \
    --loss 'CPR' \
    --savepath 'checkpoint/ICON/ICON-V/' \
    --valid True
!

:<<!
echo "Now Training ICON-R..."
python main/train.py \
    --model 'ICON-R' \
    --dataset 'datasets/SOC/Train' \
    --lr 0.01 \
    --decay 5e-4 \
    --momen 0.9 \
    --batchsize 32 \
    --loss 'CPR' \
    --savepath 'checkpoint/ICON/ICON-R-SOC/' \
    --valid True\
!
