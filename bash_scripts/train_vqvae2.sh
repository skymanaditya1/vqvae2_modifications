#!/bin/bash 

#SBATCH --job-name=temp
#SBATCH --mem-per-cpu=1024
#SBATCH --partition long
#SBATCH --account cvit_bhaasha
#SBATCH --gres=gpu:4
#SBATCH --mincpus=40
#SBATCH --nodes=1
#SBATCH --time 4-00:00:00
#SBATCH --signal=B:HUP@600
#SBATCH -w gnode80

source /home2/aditya1/miniconda3/bin/activate stargan-v2
cd /ssd_scratch/cvit/aditya1/vq-vae-2-pytorch
python train_vqvae.py /ssd_scratch/cvit/aditya1/CelebAPaired/ --n_gpu 4 --epoch 1000
