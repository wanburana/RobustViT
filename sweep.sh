edge_candidates=(0.1 1.0 1.5 2.0)
texture_candidates=(0.1 1.0 1.5 2.0)
background_candidates=(0.3 0.5 1.0 2.0)
segmentation_lamb=0.8
classification_lamb=0.2

for edge_lamb in ${edge_candidates[@]}; do
    for texture_lamb in ${texture_candidates[@]}; do
        for background_lamb in ${background_candidates[@]}; do
            echo $edge_lamb $texture_lamb $background_lamb
            
            PATH_TO_SEGMENTATION_DATA=~/work/data/general/imagenet-s/ImageNetS919/train-semi-segmentation
            PATH_TO_EDGE_DATA=~/work/data/general/generated_edges/test
            PATH_TO_IMAGENET=~/Dataset/CV/imagenet/train
            LR=3e-6

            SEG=0.8
            ACC=0.2
            BACK=$background_lamb
            EDGE=$edge_lamb
            TEXTURE=$texture_lamb

            python3 imagenet_finetune.py --seg_data $PATH_TO_SEGMENTATION_DATA --edge_data $PATH_TO_EDGE_DATA --data $PATH_TO_IMAGENET --gpu 0  --lr $LR --lambda_seg $SEG --lambda_acc $ACC --lambda_background $BACK --lambda_edge $EDGE --lambda_texture $TEXTURE  
        done
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
