import os
import glob 

import torch
import numpy as np

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AMASS_DS(Dataset):
    def __init__(self, data_path):
        self.data = np.load(data_path)

    def __len__(self):
       return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



ds = AMASS_DS('data/amass_processed/train_split.npy')