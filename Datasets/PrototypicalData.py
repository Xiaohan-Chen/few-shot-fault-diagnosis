"""
Data: 2022/11/11
Author: Xiaohan Chen
Mail: cxh_bb@outlook.com
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import random

class ProtitypicalData(Dataset):
    def __init__(self, dataset):
        super(ProtitypicalData, self).__init__()
        self.dataset = dataset
        self.classes = len(dataset)
        self.x, self.y = self.read(dataset)
    
    def read(self, dataset):
        x = np.concatenate([dataset[key] for key in dataset.keys()])
        y = []
        for i, key in enumerate(dataset.keys()):
            number = len(dataset[key])
            y.append(np.tile(i, number))
        y = np.concatenate(y)
        return x, y
    
    def __len__(self):
        count = 0
        for key in self.dataset.keys():
            count += len(self.dataset[key])
        return count

    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        # transform array to tensor
        data = torch.from_numpy(data).float()
        label = torch.tensor(label, dtype=torch.float)
        return data, label