import os
import glob 

import torch
import numpy as np

from torch.utils.data import Dataset


class AMASS_DS(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)
        self.poses = self.data['pose_body']
        self.motion_freq = self.data['motion_freq']
        self.num_joints = 21

    def __len__(self):
       return len(self.poses)

    def __getitem__(self, idx):
        return self.poses[idx].astype(np.float32)
