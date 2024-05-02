#!/bin/bash
#pythonname='/GPFS/data/hanchongyan/pytorch_1.2.0a0+8554416-py36tf'

dataname='/GPFS/data/hanchongyan-1/BRATS2020'
#pypath=$pythonname
#cudapath='/GPFS/data/hanchongyan/cuda-9.0'
datapath=${dataname}_Training_none_npy
savepath='/GPFS/data/hanchongyan-1/output_frozen.v2_r=4_2020'
#savepath='./output' 

export CUDA_VISIBLE_DEVICES=6

#export PATH=$cudapath/bin:$PATH
#export LD_LIBRARY_PATH=$cudapath/lib64:$LD_LIBRARY_PATH
#PYTHON=$pypath/bin/python3.6
#export PATH=$pypath/include:$pypath/bin:$PATH
#export LD_LIBRARY_PATH=$pypath/lib:$LD_LIBRARY_PATH

#resume='/GPFS/data/hanchongyan-1/output_sharetrans.v2_r=4_2020/model_last.pth'
#python train.py --batch_size=2 --datapath $datapath --savepath $savepath --num_epochs 1000 --dataname 'BRATS2020' #--resume $resume

#eval:

resume='/GPFS/data/hanchongyan-1/output_frozen.v2_r=4_2020/model_last.pth'
python train.py --batch_size=1 --datapath $datapath --savepath $savepath --num_epochs 0 --dataname 'BRATS2020' --resume $resume
