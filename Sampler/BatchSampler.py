"""
Batch sampler: yield a batch of indexes at each episode.
Data: 2022/11/11
Author: Xiaohan Chen
Mail: cxh_bb@outlook.com
Reference: https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/src/prototypical_batch_sampler.py
"""

import numpy as np
import torch

class BatchSampler(object):
    def __init__(self, labels, support, query, episodes):
        '''
        Initialize the BatchSampler object
        Args:
        - labels: an episode contains all labels
        - support: number of support samples per class
        - querry: number of querry samples per class
        - episodes: number of episodes per epoch
        '''
        super(BatchSampler, self).__init__()
        self.labels = labels
        self.support = support
        self.query = query
        self.episodes = episodes
        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)
        self.num_per_class = self.counts[0] # the numbers per class are equal
        self.index_matrix = self.create_matrix(self.labels, len(self.classes), self.num_per_class)
    
    def create_matrix(self, labels, num_class, num_per_class):
        '''
        Creat an index matrix, dim: classes x samples per class.
        Every class contains the same number of samples.
        args:
        - labels: an episode contains all labels
        - num_classes: number of classes
        - num_per_class: number of elements per class
        '''
        num_samples = len(labels)
        index_matrix = torch.arange(num_samples).reshape(num_class, num_per_class)
        return index_matrix
    
    def __len__(self):
        return self.episodes
    
    def __iter__(self):
        '''
        Yield a batch of indexes.
        Every batch contains the same classes but in random order.
        Every class contains (support + query) samples.
        '''
        num_class = len(self.classes)
        num_item = self.support + self.query
        batch_size = num_class * num_item
        for ep in range(self.episodes):
            batch = torch.LongTensor(batch_size)
            class_index = torch.randperm(num_class) # random the class index
            for i, label in enumerate(class_index):
                s = slice(i*num_item, (i+1)*num_item)
                sample_index = torch.randperm(self.num_per_class)[:num_item] # select element indexes from every class
                batch[s] = self.index_matrix[label][sample_index]
            yield batch
            



