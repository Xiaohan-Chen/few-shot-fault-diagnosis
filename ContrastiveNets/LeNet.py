'''
Reference: https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
'''

import torch
import torch.nn as nn

#Defining the convolutional neural network
class LeNet5(nn.Module):
    def __init__(self, dim=128, norm_layer=None):
        super(LeNet5, self).__init__()
        self.norm_layer = norm_layer if norm_layer else nn.BatchNorm1d
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 6, kernel_size=5, stride=1, padding=0),
            self.norm_layer(6),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=5, stride=1, padding=0),
            self.norm_layer(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(4048, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, dim)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out