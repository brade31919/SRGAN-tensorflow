#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./val_result/SRGAN_VGG54/ \
    --summary_dir ./val_result/SRGAN_VGG54/log/ \
    --mode test \
    --is_training False \
    --task SRGAN \
    --batch_size 16 \
    --input_dir_LR ./data/Set14_LR/ \
    --input_dir_HR ./data/Set14_HR/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --pre_trained_model True \
    --checkpoint ./experiment_SRGAN_VGG54/model-200000

