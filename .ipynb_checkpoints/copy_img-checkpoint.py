import os
import math
import argparse
import random
import logging

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from data.data_sampler import DistIterSampler

import options.options as option
from utils import util
from data import create_dataloader, create_dataset
from models import create_model

from shutil import copy

origin_path = "/data/chengzhicao/NTURain/NTURain/JRSRD/gt_total"
origin_out_path = "/data/chengzhicao/NTURain/NTURain/JRSRD/gt_few"


if not os.path.isdir(origin_out_path):
    os.makedirs(origin_out_path)
            
img_list = os.listdir(origin_path)
img_list.sort()

for i in range(len(img_list)):
    if i % 10 == 0:
        img_path = os.path.join(origin_path,img_list[i])
        out_path = os.path.join(origin_out_path,img_list[i])
        copy(img_path,out_path)

        