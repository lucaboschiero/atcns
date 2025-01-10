from __future__ import print_function

import numpy as np
import torch


class Partition(torch.utils.data.Dataset):
    """ 
    Dataset-like object, but only access a subset of it. 
    This is useful for splitting the data into partitions for different clients.
    """

    def __init__(self, data, index):
        self.data = data              # The entire dataset
        self.index = index            # Indices for the subset this Partition will access
        self.classes = 0              # Placeholder for the number of classes (set later)

    def __len__(self):
        return len(self.index)          # Length of the partition (subset)

    def __getitem__(self, i):
        data_idx = self.index[i]           # Get the index for this partition
        return self.data[data_idx]         # Return the corresponding data point


class customDataLoader():
    """ Virtual class: load a particular partition of dataset
        Class that splits the dataset into partitions and provides a DataLoader for each partition.
        This is the parent class from which iidLoader, byLabelLoader and dirichletLoader classes inherit from
    """

    def __init__(self, size, dataset, bsz):
        '''
        size: number of paritions in the loader (i.e. number of clients)
        dataset: pytorch dataset
        bsz: batch size of the data loader
        '''
        self.size = size            # Number of clients/partitions
        self.dataset = dataset
        self.classes = np.unique(dataset.targets).tolist()        # List of unique classes
        self.bsz = bsz
        self.partition_list = self.getPartitions()           # Get partitions using a custom method

        # Check that all data points are assigned to a partition
        num_unique_items = len(np.unique(np.concatenate(self.partition_list)))  
        if (len(dataset) != num_unique_items):
            print(
                f"Number of unique items in partitions ({num_unique_items}) is not equal to the size of dataset ({len(dataset)}), some data may not be included")

    def getPartitions(self):
        raise NotImplementedError("Subclasses must implement this method")

    def __len__(self):
        return self.size

    def __getitem__(self, rank):
        """
        Returns a DataLoader for a specific partition (subset) identified by 'rank'.
        """
        assert rank < self.size, 'partition index should be smaller than the size of the partition'
        partition = Partition(self.dataset, self.partition_list[rank])        # Create a Partition object
        partition.classes = self.classes                                      # Set the number of classes

        # Create a DataLoader for this partition
        train_set = torch.utils.data.DataLoader(partition, batch_size=int(self.bsz), shuffle=True,
                                                drop_last=True)  # drop last since some network requires batchnorm
        return train_set


class iidLoader(customDataLoader):
    """
    Splits the dataset into IID partitions (randomly and evenly distributed).
    """

    def __init__(self, size, dataset, bsz=128):
        super(iidLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        """
        Randomly partitions the dataset into equal-sized chunks.
        """
        data_len = len(self.dataset)
        indexes = [x for x in range(0, data_len)]          # Create a list of indices
        np.random.shuffle(indexes)                         # Shuffle the indices

        # fractions of data in each partition
        partition_sizes = [1.0 / self.size for _ in range(self.size)]

        partition_list = []
        for frac in partition_sizes:
            part_len = int(frac * data_len)              # Length of each partition
            partition_list.append(indexes[0:part_len])   # Add partition to list
            indexes = indexes[part_len:]                 # Remove used indices
        return partition_list


class byLabelLoader(customDataLoader):
    """
    Splits the dataset such that each partition contains data from a single class.
    """
    def __init__(self, size, dataset, bsz=128):
        super(byLabelLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)

        partition_list = []
        self.labels = np.unique(self.dataset.targets).tolist()               # Unique class labels
        label = self.dataset.targets
        label = torch.tensor(np.array(label))
        for i in self.labels:
            label_iloc = (label == i).nonzero(as_tuple=False).squeeze().tolist()        # Find indices of data points with label 'i'
            partition_list.append(label_iloc)                               # Append indices to partition list
        return partition_list


class dirichletLoader(customDataLoader):
    """
    Partitions the dataset using a **Dirichlet distribution** to create imbalanced partitions.
    """
    def __init__(self, size, dataset, alpha=0.9, bsz=128):
        # alpha is used in getPartition,
        # and getPartition is used in parent constructor
        # hence need to initialize alpha first
        self.alpha = alpha        # Dirichlet parameter
        super(dirichletLoader, self).__init__(size, dataset, bsz)

    def getPartitions(self):
        data_len = len(self.dataset)

        partition_list = [[] for j in range(self.size)]
        self.labels = np.unique(self.dataset.targets).tolist()
        label = self.dataset.targets
        label = torch.tensor(np.array(label))
        for i in self.labels:
            label_iloc = (label == i).nonzero(as_tuple=False).squeeze().numpy()
            np.random.shuffle(label_iloc)
            p = np.random.dirichlet([self.alpha] * self.size)
            # choose which partition a data is assigned to
            assignment = np.random.choice(range(self.size), size=len(label_iloc), p=p.tolist())
            part_list = [(label_iloc[(assignment == k)]).tolist() for k in range(self.size)]
            for j in range(self.size):
                partition_list[j] += part_list[j]
        return partition_list


if __name__ == '__main__':
    from torchvision import datasets, transforms

    dataset = datasets.MNIST('./data',
                             train=True,
                             download=True,
                             transform=transforms.Compose([transforms.ToTensor(),
                                                           transforms.Normalize((0.1307,), (0.3081,))]))
    loader = iidLoader(10, dataset)
    print(f"\nInitialized {len(loader)} loaders, each with batch size {loader.bsz}.\
    \nThe size of dataset in each loader are:")
    print([len(loader[i].dataset) for i in range(len(loader))])
    print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    loader = byLabelLoader(10, dataset)
    print(f"\nInitialized {len(loader)} loaders, each with batch size {loader.bsz}.\
    \nThe size of dataset in each loader are:")
    print([len(loader[i].dataset) for i in range(len(loader))])
    print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")

    loader = dirichletLoader(10, dataset, alpha=0.9)
    print(f"\nInitialized {len(loader)} loaders, each with batch size {loader.bsz}.\
    \nThe size of dataset in each loader are:")
    print([len(loader[i].dataset) for i in range(len(loader))])
    print(f"Total number of data: {sum([len(loader[i].dataset) for i in range(len(loader))])}")
