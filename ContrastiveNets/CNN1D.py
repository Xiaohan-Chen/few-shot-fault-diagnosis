import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, dim=128, norm_layer=None):
        super(CNN1D, self).__init__()
        self.norm_layer = norm_layer if norm_layer else nn.BatchNorm1d
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=3,padding=1),
            self.norm_layer(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
            )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=3,padding=1),
            self.norm_layer(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,padding=1),
            self.norm_layer(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
            )
        self.avgpool = nn.AdaptiveAvgPool1d(64) # output (batch, 64, 64)
        self.fc = nn.Sequential(nn.Linear(64*64,dim))

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)  # [200, 64, 128]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x