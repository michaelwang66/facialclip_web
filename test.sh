#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --qos=long-high-prio 
#SBATCH --time=7-0:0:0
#SBATCH -p res-gpu-small
# SBATCH --mem=28g #Maximum 28
#SBATCH --output=testdfew
#SBATCH --gres=gpu:pascal:1

python test.py \
--dataset 'DFEW' \
--exper-name '2508291647mydfewexp' \
--seed 123 \
--contexts-number 8 \
--class-specific-contexts 'True' \
--text-type 'class_descriptor'
