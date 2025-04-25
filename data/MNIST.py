"""
@Author: hzx
@Date: 2025-04-25
@Version: 1.0
"""

import torch
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


def load_mnist(batch_size: int):
    transform = ToTensor()
    train_mnist = MNIST(root='/home/hzx/workspace/dataset', download=True, transform=transform,
                        train=True)
    test_mnist = MNIST(root='/home/hzx/workspace/dataset', download=True, transform=transform,
                       train=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_mnist)
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_mnist)
    train_mnist_data_loader = DataLoader(train_mnist, batch_size=batch_size, shuffle=False, sampler=train_sampler)
    test_mnist_data_loader = DataLoader(test_mnist, batch_size=batch_size, shuffle=False, sampler=test_sampler)
    return train_mnist_data_loader, test_mnist_data_loader
