# pip install torch torchvision xarray netCDF4

from math import ceil
from sys import stdout

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import xarray as xr

def print_progress(progression_str, current, end):
    stdout.write('\r')
    stdout.write(progression_str.format(current, end))
    stdout.flush()

batch_size = 256

transform = transforms.Compose([
    lambda img: np.array(img)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

train_dataset = xr.Dataset()
test_dataset = xr.Dataset()
train_container = xr.DataArray(np.zeros((10, 5000, 32, 32, 3), dtype='u1'))
test_container = xr.DataArray(np.zeros((10, 1000, 32, 32, 3), dtype='u1'))

train_batch_cnt = ceil(50000/batch_size)
test_batch_cnt = ceil(10000/batch_size)

progression_str = 'Creating training dataset: batch {:d}/{:d}'
stdout.write(progression_str.format(0, train_batch_cnt))
stdout.flush()

train_indices = [0] * 10
for batch_idx, (inputs, targets) in enumerate(trainloader):
    for idx, (input, target) in enumerate(zip(inputs, targets)):
        class_idx = target.item()
        img_idx = train_indices[class_idx]
        train_container[class_idx, img_idx] = input.cpu().numpy()
        train_indices[class_idx] += 1
    print_progress(progression_str, batch_idx+1, train_batch_cnt)

train_dataset['data'] = train_container
train_dataset.to_netcdf('./data/train.h5')

progression_str = 'Creating test dataset: batch {:d}/{:d}'
stdout.write('\n'+progression_str.format(0, test_batch_cnt))
stdout.flush()

test_indices = [0] * 10
for batch_idx, (inputs, targets) in enumerate(testloader):
    for idx, (input, target) in enumerate(zip(inputs, targets)):
        class_idx = target.item()
        img_idx = test_indices[class_idx]
        test_container[class_idx, img_idx] = input.cpu().numpy()
        test_indices[class_idx] += 1
    print_progress(progression_str, batch_idx+1, test_batch_cnt)
stdout.write('\n')
stdout.flush()

test_dataset['data'] = test_container
test_dataset.to_netcdf('./data/test.h5')
