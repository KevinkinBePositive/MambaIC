#!/bin/bash
#SBATCH -J train_MambaIC                              # 作业名称  
#SBATCH -N 1                                   # 使用一个节点  
#SBATCH -c 4                                   # 每个任务的 CPU 核心数     
#SBATCH --gres=gpu:1                            # 请求分配两个 GPU  
#SBATCH -w inspur0                              # 指定在 inspur0 节点上运行
#SBATCH -o train_MambaIC_%j.out   # 标准输出文件  
#SBATCH -e train_MambaIC_%j.err   # 标准错误输出文件   


SCRIPT_PATH=/home/$USER/MambaIC_cursor/mambaic/
data_path=/data/$USER/flickr30k
num_epoch=500
batch_size=12
checkpoint_path=/data/$USER/mambaic_output_0630_2


CUDA_VISIBLE_DEVICES=3 python3 $SCRIPT_PATH/train.py --cuda -d=$data_path  -n 128 --lambda 0.05 --epochs=$num_epoch --lr_epoch 450 490 --batch-size=$batch_size --save_path=$checkpoint_path  --save
# CUDA_VISIBLE_DEVICES=3, 4 python3 -m torch.distributed.launch --nproc_per_node=2 $SCRIPT_PATH/train.py --cuda -d=$data_path -n 128 --lambda 0.05 --epochs=$num_epoch --lr_epoch 450 490 --batch-size=$batch_size--save_path=$checkpoint_path --save