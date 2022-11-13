import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(1,32,kernel_size=3,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
            )
        self.layer2 = nn.Sequential(
            nn.Conv1d(32,64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
            )
        self.layer3 = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, padding=0)
            )
        self.avgpool = nn.AdaptiveAvgPool1d(1) # output (64,1)
        self.fc = nn.Sequential(nn.Linear(64,32), nn.Sigmoid())
        self.out = nn.Linear(32,1)

    def forward_one(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

    def forward(self, x1, x2):
        out1 = self.forward_one(x1)
        out2 = self.forward_one(x2)
        dis = torch.abs(out1 - out2)
        out = self.out(dis)
        return out