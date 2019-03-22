import torch 
from datagen import MyDataset
import os
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np
from torch.utils.data import DataLoader
import torchvision.models
import torch.nn as nn
import torch.optim as optim
#import matplotlib.pyplot as plt


def load_split_train_test(data_dir, valid_size = .2):

    cur_dir = os.path.dirname(__file__)
    datadir = os.path.join(cur_dir, data_dir)

    normalize=transforms.Normalize(mean=[.5,.5,.5],std=[.5,.5,.5])
    train_transforms = transforms.Compose([transforms.Resize(224),
                                       transforms.ToTensor(),
                                       normalize
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(224),
                                      transforms.ToTensor(),
                                      normalize
                                      ])
    train_data = torchvision.datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = torchvision.datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    from torch.utils.data.sampler import SubsetRandomSampler
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=1)
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=1)
    return trainloader, testloader
