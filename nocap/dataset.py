import os
import glob 

import torch
import numpy as np

from torch.utils.data import Dataset


class NoCAP_DS(Dataset):
    def __init__(self, data_path, disc=False):
        self.data = np.load(data_path, allow_pickle=True)
        self.poses = self.data['pose_body']
        self.motion_freq = self.data['motion_freq']
        self.betas = self.data.get('betas', None)
        self.root_orient = self.data.get('root_orient', None)
        self.codebook = self.data.get('codebook', None)
        if self.codebook is not None:
            self.codebook = self.codebook[()]

        self.num_joints = 21
        self.disc = disc

    def __len__(self):
       return len(self.poses)

    def __getitem__(self, idx):
        pose = self.poses[idx]
        if not self.disc:
            pose = pose.astype(np.float32)
        return pose
