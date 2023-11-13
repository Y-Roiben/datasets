import torchvision
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np


def get_cinic(merge=False):
    cinic_dir = r"~/dataset/CINIC-10"
    cinic_mean = [0.47889522, 0.47227842, 0.43047404]
    cinic_std = [0.24205776, 0.23828046, 0.25874835]

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=1),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(cinic_mean, cinic_std),
    ])
    trainset = torchvision.datasets.ImageFolder(cinic_dir + '/train', transform=transform_train)

    testset = torchvision.datasets.ImageFolder(cinic_dir + '/test', transform=transform_test)

    validset = torchvision.datasets.ImageFolder(cinic_dir + '/valid', transform=transform_train)

    if merge:
        trainset = trainset + validset

    return trainset, testset


