import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import time
import random

class SiameseTrain(Dataset):
    def __init__(self, dataset):
        super(SiameseTrain, self).__init__()
        np.random.seed(29)
        self.dataset = dataset
        self.num_classes = len(dataset)
    
    def __len__(self):
        count = 0
        for key in self.dataset.keys():
            count += len(self.dataset[key])
        return count
    
    def __getitem__(self, index):
        # get sample pair from the same classes
        if index % 2 == 1:
            label = 1
            idx1 = random.randint(0, self.num_classes-1)
            x1 = random.choice(self.dataset[idx1])
            x2 = random.choice(self.dataset[idx1])
        else:
            label = 0
            idx1 = random.randint(0, self.num_classes-1)
            idx2 = random.randint(0, self.num_classes-1)
            while idx1 == idx2:
                idx2 = random.randint(0, self.num_classes-1)
            x1 = random.choice(self.dataset[idx1])
            x2 = random.choice(self.dataset[idx2])
        
        # transform array to tensor
        x1, x2 = torch.from_numpy(x1).float(), torch.from_numpy(x2).float()
        label = torch.tensor(label, dtype=torch.float)
        
        return x1, x2, label

class SiameseTest(Dataset):
    def __init__(self, dataset):
        super(SiameseTest).__init__()
        np.random.seed(29)
        self.dataset = dataset
        self.num_classes = len(dataset)
        self.label1 = None
        self.x1 = None

    def __len__(self):
        count = 0
        for key in self.dataset.keys():
            count += len(self.dataset[key])
        return count
    
    def __getitem__(self, index):
        idx = index % self.num_classes
        # get sample pair from same class
        if idx ==0:
            self.label1 = random.randint(0, self.num_classes-1)
            x1 = random.choice(self.dataset[self.label1])
            x2 = random.choice(self.dataset[self.label1])
        # generate sample pair from different class
        else:
            label2 = random.randint(0, self.num_classes-1)
            x1 = random.choice(self.dataset[self.label1])
            while self.label1 == label2:
                label2 = random.randint(0, self.num_classes-1)
            x2 = random.choice(self.dataset[label2])

        # transform array to tensor
        x1, x2 = torch.from_numpy(x1).float(), torch.from_numpy(x2).float()
        
        return x1, x2
