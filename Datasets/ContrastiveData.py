"""
Data: 2022/11/19
Author: Xiaohan Chen
Mail: cxh_bb@outlook.com
"""

import torch
from torch.utils.data import Dataset
from Datasets.Transforms import random_waveform_transforms, waveform_transforms_test
import numpy as np
import time


class ContrastiveData(Dataset):
    def __init__(self, dataset):
        '''
        dataset is a dictionary, labels are keys, values are the corresponding fault signals
        '''
        super(ContrastiveData, self).__init__()
        self.dataset = dataset
        self.classes = len(dataset)
        self.x, self.y = self.read(dataset)
    
    def read(self, dataset):
        '''
        convert the dictionary to ndarray
        '''
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
        data = waveform_transforms_test(data)
        label = torch.tensor(label, dtype=torch.float)
        return data, label

class ContrastivePair(ContrastiveData):
    '''
    generative signal pair with random waveform transformations
    Reference: https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb
    '''
    def __getitem__(self, index):
        data = self.x[index]
        label = self.y[index]

        # transform array to tensor
        data1 = random_waveform_transforms(data)
        data2 = random_waveform_transforms(data)

        return data1, data2

