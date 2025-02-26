from __future__ import print_function
import json
import os
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from dataloader import *  # Ensure this correctly handles FEMNIST partitions
from PIL import Image
import matplotlib.pyplot as plt
import random

# Define the network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Adjusted feature size after convolutions and pooling
        self.fc1 = nn.Linear(64 * 6 * 6, 256)  # Adjust based on final feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 62)  # FEMNIST has 62 classes (0-9, a-z, A-Z)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        
        x = x.view(-1, self.num_flat_features(x))  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:])).item()


class FEMNISTDataset(Dataset):
    def __init__(self, data_path, split="train", transform=None, num_clients=None):
        self.data_path = os.path.join(data_path, split)
        self.transform = transform
        self.images = []
        self.labels = []
        self.client_ids = []
        self.client_data = {}

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Directory {self.data_path} not found!")

        # Read all clients' data
        for file_name in os.listdir(self.data_path):
            if file_name.endswith(".json"):
                file_path = os.path.join(self.data_path, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)

                for user in data['users']:
                    for x, y in zip(data['user_data'][user]['x'], data['user_data'][user]['y']):
                        img = torch.tensor(x, dtype=torch.float32).view(28, 28)  
                        self.images.append(img)
                        self.labels.append(y)
                        self.client_ids.append(user)

                        if user not in self.client_data:
                            self.client_data[user] = []
                        self.client_data[user].append(len(self.images) - 1)

        self.targets = torch.tensor(self.labels, dtype=torch.long)

        if num_clients:
            self._partition_data(num_clients)

    def _partition_data(self, num_clients):
        """
        Selects data only from the first 'num_clients' clients.
        """
        available_clients = list(self.client_data.keys())

        # Ensure we only take the first 'num_clients' clients
        selected_clients = random.sample(available_clients, num_clients)

        # Flatten list of indices for selected clients
        selected_indices = [idx for client in selected_clients for idx in self.client_data[client]]

        # Filter dataset to only include selected clients' data
        self.images = [self.images[i] for i in selected_indices]
        self.labels = [self.labels[i] for i in selected_indices]
        self.client_ids = [self.client_ids[i] for i in selected_indices]
        
        # Update target labels
        self.targets = torch.tensor(self.labels, dtype=torch.long)


    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        img = Image.fromarray(img.numpy())

        if self.transform:
            img = self.transform(img)

        return img, label

def getFEMNISTDataset(split="train", num_clients=None):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure consistency with model input
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return FEMNISTDataset('./leaf/data/femnist/data', split=split, transform=transform, num_clients=num_clients)


def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/femnist_loader.pk'):
    assert loader_type in ['iid', 'byLabel', 'dirichlet', 'femnist'], 'Invalid loader type'
    
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
        except FileNotFoundError:
            print('Loader not found, initializing a new one...')
            dataset = getFEMNISTDataset("train", num_clients=1000) 
            loader = loader_type(num_clients, dataset)
    else:
        print('Initializing a new data loader...')
        dataset = getFEMNISTDataset("train", num_clients=1000) 
        loader = loader_type(num_clients, dataset)

    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)

    return loader

def test_dataloader(test_batch_size):
    dataset = getFEMNISTDataset("test")
    img, label = dataset[0]  # Get the first image and label

    """plt.imshow(img.squeeze(), cmap="gray")  # Remove extra dimensions and set grayscale colormap
    plt.title(f"Label: {label}")
    plt.show()"""
    return DataLoader(dataset, batch_size=test_batch_size, shuffle=False)


if __name__ == '__main__':
    from torchsummary import summary
    
    print("# Initialize FEMNIST network")
    net = Net()
    summary(net.cuda(), (1, 32, 32))  # Matches the transformation resize
    
    print("\n# Initialize FEMNIST dataloaders")
    loader_types = ['iid', 'byLabel', 'dirichlet']
    for loader_type in loader_types:
        loader = train_dataloader(10, loader_type, store=False)
        print(f"Initialized {len(loader)} loaders (type: {loader_type}), each with batch size {loader.bsz}.")
        print("Dataset sizes:", [len(loader[i].dataset) for i in range(len(loader))])
        print(f"Total samples: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    print("\n# Feeding data to network")
    x = next(iter(loader[0]))[0].cuda()
    y = net(x)
    print(f"Input shape: {x.shape}, Output shape: {y.shape}")
