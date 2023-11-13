from torch.utils.data import DataLoader
from torchvision.datasets import GTSRB
from torchvision import transforms
import torch
import numpy


def get_GTSRB():
    train_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.30387896, 0.30411962, 0.30566233),
                             (0.27668217, 0.27587897, 0.2751373))
    ])
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.3403, 0.3121, 0.3214),
                             (0.2724, 0.2608, 0.2669))
    ])
    test_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.30177543, 0.30202687, 0.303099),
                             (0.2756153, 0.2754654, 0.27469474))
    ])
    trainset = GTSRB(root=r"C:\Users\hp\dataset", split="train", download=True, transform=train_transform)
    testset = GTSRB(root=r"C:\Users\hp\dataset", split="test", download=True, transform=test_transform)
    return trainset, testset


def get_mean_std(dataset):
    # Var[x] = E[X**2]-E[X]**2
    train_loader = DataLoader(dataset=dataset, batch_size=64, shuffle=False)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for data, _ in train_loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    print(num_batches)
    print(channels_sum)
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


def compute_mean_std(dataset):
    """compute the mean and std of dataset

    Returns:
        a tuple contains mean, std value of entire dataset
    """

    data_r = numpy.dstack([dataset[i][0][:, :, 0] for i in range(len(dataset))])
    data_g = numpy.dstack([dataset[i][0][:, :, 1] for i in range(len(dataset))])
    data_b = numpy.dstack([dataset[i][0][:, :, 2] for i in range(len(dataset))])
    mean = numpy.mean(data_r), numpy.mean(data_g), numpy.mean(data_b)
    std = numpy.std(data_r), numpy.std(data_g), numpy.std(data_b)

    return mean, std


if __name__ == "__main__":
    transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor(), ])
    trainset = GTSRB(root=r"C:\Users\hp\dataset", split="train", download=True, transform=transform)
    testset = GTSRB(root=r"C:\Users\hp\dataset", split="test", download=True, transform=transform)
    train_mean, train_std = compute_mean_std(trainset)
    test_mean, test_std = compute_mean_std(testset)
    print(train_mean, train_std)
    print(test_mean, test_std)
