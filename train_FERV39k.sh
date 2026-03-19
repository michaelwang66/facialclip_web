#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --qos=long-high-prio 
#SBATCH --time=7-0:0:0
#SBATCH -p res-gpu-small
# SBATCH --mem=28g #Maximum 28
#SBATCH --output=trainfer
#SBATCH --gres=gpu:ampere:1

python main.py \
--dataset 'FERV39K' \
--workers 16 \
--epochs 50 \
--exper-name myexp \
--batch-size 12 \
--lr 1e-2 \
--lr-image-encoder 1e-5 \
--lr-prompt-learner 1e-3 \
--weight-decay 1e-4 \
--momentum 0.9 \
--print-freq 10 \
--milestones 30 40 \
--contexts-number 8 \
--class-token-position "end" \
--class-specific-contexts 'True' \
--text-type 'class_descriptor' \
--seed 123 \
--temporal-layers 1 \