from __future__ import print_function

import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
from torchvision.models.resnet import resnet18

from dataloader import *

# Definition of the neural network model
"""class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 input image channels (RGB), 6 output channels, 3x3 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 3)  # Changed from 1 to 3 channels
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension after pooling
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # After conv1, input becomes (3, 32, 32) -> (6, 30, 30)
        # Second convolution layer, followed by max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # After conv2, input becomes (6, 30, 30) -> (16, 28, 28) -> (16, 14, 14)
        x = x.view(-1, self.num_flat_features(x))  # Flatten the output for the fully connected layers
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features"""

def Net():
    num_classes = 10
    model = resnet18(pretrained=True)
    n = model.fc.in_features
    model.fc = nn.Linear(n, num_classes)
    return model
    """def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3,   64,  3)
        self.conv2 = nn.Conv2d(64,  128, 3)
        self.conv3 = nn.Conv2d(128, 256, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.drop1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.drop2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, 256)
        self.drop3 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.drop1(x)
        x = F.relu(self.fc1(x))
        x = self.drop2(x)
        x = F.relu(self.fc2(x))
        x = self.drop3(x)
        x = self.fc3(x)
        return x"""


def getDataset():
    dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
]))
    return dataset


def basic_loader(num_clients, loader_type):
    dataset = getDataset()
    return loader_type(num_clients, dataset)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/loader.pk'):
    assert loader_type in ['iid', 'byLabel',
                           'dirichlet'], 'Loader has to be one of the  \'iid\',\'byLabel\',\'dirichlet\''
    if loader_type == 'iid':
        loader_type = iidLoader
    elif loader_type == 'byLabel':
        loader_type = byLabelLoader
    elif loader_type == 'dirichlet':
        loader_type = dirichletLoader

    if store:
        try:
            with open(path, 'rb') as handle:
                loader = pickle.load(handle)
        except:
            print('loader not found, initialize one')
            loader = basic_loader(num_clients, loader_type)
    else:
        print('initialize a data loader')
        loader = basic_loader(num_clients, loader_type)
    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader


def test_dataloader(test_batch_size):
    test_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10('./data', train=False, transform=transforms.Compose([transforms.ToTensor(),
                                                                              transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))])),
        batch_size=test_batch_size, shuffle=True)
    return test_loader


if __name__ == '__main__':
    from torchsummary import summary

    print("#Initialize a network")
    net = Net()
    summary(net.cuda(), (3, 32, 32))

    print("\n#Initialize dataloaders")
    loader_types = ['iid', 'dirichlet']
    for i in range(len(loader_types)):
        loader = train_dataloader(50, loader_types[i], store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_types[i]}), each with batch size {loader.bsz}.\
        \nThe size of dataset in each loader are:")
        print([len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n#Feeding data to network")
    x = next(iter(loader[i]))[0].cuda()
    y = net(x)
    print(f"Size of input:  {x.shape} \nSize of output: {y.shape}")