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

# Define the network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)  
        self.conv2 = nn.Conv2d(32, 64, 3)
        
        # Adjusted feature size after convolutions and pooling
        self.fc1 = nn.Linear(64 * 5 * 5, 256)  # Adjust based on final feature map size
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)  # FEMNIST has 62 classes (0-9, a-z, A-Z)
    
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
    def __init__(self, data_path, split="train", transform=None):
        self.data_path = data_path
        self.split = split
        self.transform = transform

        # Carica i dati dal file JSON
        with open(os.path.join(data_path, f"{split}.json"), 'r') as f:
            self.data = json.load(f)

        # Filtra solo i numeri (etichette 0-9)
        self.images = []
        self.labels = []
        for user in self.data['users']:
            for x, y in zip(self.data['user_data'][user]['x'], self.data['user_data'][user]['y']):
                if 0 <= y <= 9:  # Mantieni solo i numeri
                    self.images.append(torch.tensor(x, dtype=torch.float32).view(1, 28, 28))  # Reshape
                    self.labels.append(torch.tensor(y, dtype=torch.long))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

def getFEMNISTDataset(split="train"):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),  # Ensure consistency with model input
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    return FEMNISTDataset('./leaf/data/femnist/data', split=split, transform=transform)


# Create Data Loaders for FEMNIST
def train_dataloader(num_clients, loader_type='iid', store=True, path='./data/femnist_loader.pk'):
    assert loader_type in ['iid', 'byLabel', 'dirichlet'], 'Invalid loader type'
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
            dataset = getFEMNISTDataset("train")
            loader = loader_type(num_clients, dataset)
    else:
        print('Initializing a new data loader...')
        dataset = getFEMNISTDataset("train")
        loader = loader_type(num_clients, dataset)

    if store:
        with open(path, 'wb') as handle:
            pickle.dump(loader, handle)
    
    return loader

def test_dataloader(test_batch_size):
    dataset = getFEMNISTDataset("test")
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
