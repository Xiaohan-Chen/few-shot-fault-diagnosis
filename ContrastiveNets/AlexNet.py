'''
Reference: https://blog.paperspace.com/alexnet-pytorch/
'''

import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, dim=64, norm_layer=None):
        super(AlexNet, self).__init__()
        self.norm_layer = norm_layer if norm_layer else nn.BatchNorm1d
        self.layer1 = nn.Sequential(
            nn.Conv1d(1, 96, kernel_size=11, stride=4, padding=0),
            self.norm_layer(96),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv1d(96, 256, kernel_size=5, stride=1, padding=2),
            self.norm_layer(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv1d(256, 384, kernel_size=3, stride=1, padding=1),
            self.norm_layer(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv1d(384, 384, kernel_size=3, stride=1, padding=1),
            self.norm_layer(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv1d(384, 256, kernel_size=3, stride=1, padding=1),
            self.norm_layer(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7680, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, dim))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1) # [100, 7680]
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out