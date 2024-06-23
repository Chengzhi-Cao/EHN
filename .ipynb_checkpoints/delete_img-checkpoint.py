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

origin_path = "/data/chengzhicao/NTURain/NTURain/result_real_drop"
origin_out_path = "/data/chengzhicao/NTURain/NTURain/result_real_drop_short"


if not os.path.isdir(origin_out_path):
    os.makedirs(origin_out_path)

img_list = os.listdir(origin_path)
img_list.sort()

for i in range(len(img_list)):
    img_path = os.path.join(origin_path,img_list[i])
    out1_path = os.path.join(origin_out_path,img_list[i])

    if not os.path.isdir(out1_path):
        os.makedirs(out1_path)
    
    file_list = os.listdir(img_path)
    file_path = os.path.join(img_path,file_list[0])
    
    
    total_img_list = os.listdir(file_path)
    total_img_list.sort()
    
    for j in range(len(total_img_list)):
        if j % 2 == 0:
            final_img_path = os.path.join(file_path,total_img_list[j])
            print('final_img_path=',final_img_path)
            out2_path = os.path.join(out1_path,total_img_list[j])
            print('out2_path=',out2_path)

            copy(final_img_path,out2_path)

        
