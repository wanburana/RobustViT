foreground_candidates=(1.0 1.5 2.0 3.0)
background_candidates=(0.1 0.3 0.5 1.0)
segmentation_lamb=0.8
classification_lamb=0.2

for foreground_lamb in ${foreground_candidates[@]}; do
    for background_lamb in ${background_candidates[@]}; do
        echo $foreground_lamb $background_lamb
        
        PATH_TO_SEGMENTATION_DATA=~/work/data/general/generated_edges/test
        PATH_TO_IMAGENET=~/Dataset/CV/imagenet/train
        LR=3e-6

        SEG=0.8
        ACC=0.2
        BACK=$background_lamb
        FORE=$foreground_lamb

        python3 imagenet_finetune.py --seg_data $PATH_TO_SEGMENTATION_DATA --data $PATH_TO_IMAGENET --gpu 0  --lr $LR --lambda_seg $SEG --lambda_acc $ACC --lambda_background $BACK --lambda_foreground $FORE  

    done
done
# PATH_TO_SEGMENTATION_DATA=~/work/data/general/generated_edges/test
# PATH_TO_IMAGENET=~/Dataset/CV/imagenet/train
# LR=3e-6

# SEG=0.8
# ACC=0.2
# BACK=1
# FORE=1


# python3 imagenet_finetune.py --seg_data $PATH_TO_SEGMENTATION_DATA --data $PATH_TO_IMAGENET --gpu 0  --lr $LR --lambda_seg $SEG --lambda_acc $ACC --lambda_background $BACK --lambda_foreground $FORE  
