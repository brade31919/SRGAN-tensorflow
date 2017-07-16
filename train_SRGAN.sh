#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir ./experiment_SRGAN_VGG54/ \
    --summary_dir ./experiment_SRGAN_VGG54/log/ \
    --mode train \
    --is_training True \
    --task SRGAN \
    --batch_size 16 \
    --flip True \
    --random_crop True \
    --crop_size 24 \
    --input_dir_LR ./data/RAISE_LR/ \
    --input_dir_HR ./data/RAISE_HR/ \
    --num_resblock 16 \
    --perceptual_mode VGG54 \
    --name_queue_capacity 4096 \
    --image_queue_capacity 4096 \
    --ratio 0.001 \
    --learning_rate 0.0001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --max_iter 200000 \
    --queue_thread 10 \
    --vgg_scaling 0.0061 \
    --pre_trained_model True \
    --checkpoint ./experiment_SRGAN_MSE/model-500000

