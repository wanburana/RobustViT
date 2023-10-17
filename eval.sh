PATH_TO_ROBUSTNESS_DATASET=~/work/data/general/imagenet-r
BATCH_SIZE=8
PATH_TO_FINETUNED_CHECKPOINT=~/work/RobustViT/experiment/lr_3e-06_seg_0.8_acc_0.2_bckg_0.33_fgd_0.67_num_epochs_50/model_best.pth.tar

python3 imagenet_eval_robustness.py --data $PATH_TO_ROBUSTNESS_DATASET --batch-size $BATCH_SIZE --evaluate --checkpoint $PATH_TO_FINETUNED_CHECKPOINT