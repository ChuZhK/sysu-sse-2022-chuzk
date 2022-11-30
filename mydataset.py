# -*- coding: utf-8 -*-
import os
import glob
import pickle
from PIL import Image
from numpy import asarray
import numpy as np
import random
import torch.utils.data as data
from torchvision.transforms import *
import torch

class mydataset(data.Dataset):
    def __init__(self,config):
        self.data_dir = config.data_dir
        self.lmk_name = config.lmk_name
        self.before_dir = config.before_dir
        
        self.api_landmarks = pickle.load(open(os.path.join(self.data_dir, self.lmk_name), 'rb'))
         # Load the Un-makeup image path list
        self.before_list = [self.before_dir + '/' + img_path
                            for img_path in  os.listdir(os.path.join(self.data_dir , self.before_dir))]
        
    def __len__(self):
        return len(self.before_list)
    
    def __getitem__(self, index):
        before_path = self.before_list[index]
        before_lmks = self.api_landmarks[before_path].astype(np.int)
        before_img = Image.open(os.path.join(self.data_dir, before_path)).convert('RGB')
        # before_img = before_img.permute(1, 2, 0)
        before_img_arr = asarray(before_img)
        print(before_img_arr.shape)
        return torch.tensor(before_img_arr,dtype=torch.float).permute(2, 0,1)