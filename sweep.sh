PATH_TO_SEGMENTATION_DATA=~/work/data/general/generated_edges/test
PATH_TO_IMAGENET=~/Dataset/CV/imagenet/train
LR=3e-6

SEG=0.8
ACC=0.2
BACK=2
FORE=1

python3 imagenet_finetune.py --seg_data $PATH_TO_SEGMENTATION_DATA --data $PATH_TO_IMAGENET --gpu 0  --lr $LR --lambda_seg $SEG --lambda_acc $ACC --lambda_background $BACK --lambda_foreground $FORE
