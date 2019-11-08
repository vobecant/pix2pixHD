#!/bin/bash
#SBATCH --job-name=pix2pixhd_insertion
#SBATCH --output=pix2pixhd_insertion.out
#SBATCH --time=1-00:00:00
#SBATCH --mem=30GB
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=a.vobecky@gmail.com

python -u insert_random.py \
        --name label2city_1024p --netG local --ngf 32 --resize_or_crop none \
        --dataroot /home/vobecant/datasets/cityscapes/ > pix2pixhd_insertion.out
