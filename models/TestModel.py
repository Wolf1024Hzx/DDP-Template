"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

from torch import nn


class BasicBlock(nn.Module):
    def __init__(self, channels):
        super(BasicBlock, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 4, out_channels=channels * 4, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=channels * 4, out_channels=channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        features = self.sequential(x)
        return x + features


class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.sequential = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            BasicBlock(channels=16),
            BasicBlock(channels=16),
            nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=49, out_features=10)
        )

    def forward(self, x):
        return self.sequential(x)
