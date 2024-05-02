import argparse
import os
import time
import logging
import random
import numpy as np
from collections import OrderedDict

import torch
import torch.optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import mmformer
from data.transforms import *
from data.datasets_nii import Brats_loadall_nii, Brats_loadall_test_nii
from data.data_utils import init_fn
from utils import Parser,criterions
from utils.parser import setup 
from utils.lr_scheduler import LR_Scheduler, record_loss, MultiEpochsDataLoader 
from predict import AverageMeter, test_softmax
from torch.utils.data import DataLoader

train_transforms = 'Compose([RandCrop3D((128,128,128)), RandomRotion(10), RandomIntensityChange((0.1,0.1)), RandomFlip(0), NumpyType((np.float32, np.int64)),])'
test_transforms = 'Compose([NumpyType((np.float32, np.int64)),])'
datapath='/GPFS/data/hanchongyan/BRATS2020_Training_none_npy'
num_cls = 4
train_file = 'train.txt'
test_file = 'test.txt'

train_set = Brats_loadall_nii(transforms=train_transforms, root=datapath, num_cls=num_cls, train_file=train_file)
test_set = Brats_loadall_test_nii(transforms=test_transforms, root=datapath, test_file=test_file)

train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)


for index, data in enumerate(test_loader):
    
    if data is None:
        print("None found in train_loader at index:", index)
