import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import defaultdict
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
import pandas as pd

import numpy as np
from rules.correlations import C 
import matplotlib.pyplot as plt
from utils.logger import get_logger
from sklearn.decomposition import PCA

logger = get_logger()

def visualize_clusters(X, labels, centroids):
    """
    Visualize clustering results in 2D.

    Parameters:
        X (array): Dataset of shape (n_samples, n_features).
        labels (array): Cluster labels of shape (n_samples,).
        centroids (array): Centroids of shape (k, n_features).
    """
    # Reduce dimensions to 2D using PCA
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_2d = pca.fit_transform(X)
        centroids_2d = pca.transform(centroids)
    else:
        X_2d = X
        centroids_2d = centroids

    # Plot data points with cluster labels
    plt.figure(figsize=(10, 8))
    unique_labels = np.unique(labels)
    for label in unique_labels:
        plt.scatter(X_2d[labels == label, 0], X_2d[labels == label, 1],
                    label=f'Cluster {label}', s=50, alpha=0.6)
    
    # Plot centroids
    plt.scatter(centroids_2d[:, 0], centroids_2d[:, 1],
                c='red', marker='x', s=200, label='Centroids')
    
    plt.title('Clustering Results')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    plt.savefig("kmeans2.png")
    plt.show()

def k_means(input, max_k=5, max_iters=500, tol=1e-4):
    """
    Perform dynamic k-means clustering with automatic k selection.

    Parameters:
        input (torch.Tensor): Tensor of shape (1, d, n), where n is the number of points.
        max_k (int): Maximum number of clusters to consider.
        max_iters (int): Maximum iterations for k-means.
        tol (float): Tolerance for centroid movement.

    Returns:
        out (torch.Tensor): Aggregated mean of benign clusters (1, d, 1).
        attackers (list): Indices of outlier points (or attackers).
        best_k (int): Optimal number of clusters.
    """
    n = input.shape[-1]  # Number of points
    d = input.shape[1]   # Vector dimension

    if n < 2:
        raise ValueError("Not enough points for clustering.")

    inputT = deepcopy(input)

    # Convert input tensor to numpy for clustering
    X = inputT.squeeze(0).T.numpy()  # Shape: (n, d)

    # Try different values of k and calculate silhouette scores
    silhouette_scores = []
    cluster_results = {}

    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(X)

    for k in range(2,  n):
        kmeans = KMeans(n_clusters=k, n_init='auto', init='k-means++')
        labels = kmeans.fit_predict(data_scaled)
        silhouette_scores.append(silhouette_score(data_scaled, labels))
        cluster_results[k] = (kmeans.cluster_centers_, labels)
        #print("K: ", k, ", Silhouette Score: ", silhouette_score(data_scaled,labels))

    # Select the k with the best silhouette score
    best_k = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at k=2
    centroids, labels = cluster_results[best_k]

    # Identify benign clusters and attackers
    #cluster_sizes = np.array([np.sum(labels == i) for i in range(best_k)])
    #min_cluster_size = cluster_sizes.min()

    print("BEST K: ", best_k)

    #visualize_clusters(data_scaled, labels, centroids)

    cost = C(input,n)

    cluster_nodes = defaultdict(list)

    # Assuming `labels` is the array of cluster labels for each point
    for idx, label in enumerate(labels):
        cluster_nodes[label].append(idx)

    # Convert to a regular dictionary
    cluster_nodes = dict(cluster_nodes)
    
    # To store the median scores of the correlation matrix for each cluster
    cluster_medians = {}

    # Initialize empty lists to store benign clients, attackers, and temporary nodes
    benign_clients = []
    attackers = []

    # Print nodes in each cluster
    for cluster, nodes in cluster_nodes.items():
        logger.info(f"Cluster {cluster}: Nodes {nodes}")
        cluster_cost = []
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):  # Avoid duplicates (i, j) == (j, i)
                node1 = nodes[i]
                node2 = nodes[j]
                cluster_cost.append(cost[node1][node2])

        # Calculate the median of the collected values
        if cluster_cost:
            median_value = np.median(cluster_cost)
        else:
            attackers.append(node for node in nodes)  # Handle cases with no pairwise values (e.g., single node clusters)
        
        cluster_medians[cluster] = median_value

    min = +1*n
    min_cluster = None

    # Find the cluster with the minimun median, the nodes in this clusters are the benign ones
    for cluster, median in cluster_medians.items():
        logger.info(f"Cluster {cluster} median value: {median}")
        if median < min:
            min = median
            min_cluster = cluster
    
    benign_clients = cluster_nodes[min_cluster]      
    attackers.append([i for i in range(n) if i not in benign_clients])

    logger.info(f"Attackers: {attackers}")

    logger.info(f"Benign clients: {benign_clients}")

    input = input.squeeze(0)  

    out = torch.mean(input[:, [i for i in range(n) if i in benign_clients]], dim=1, keepdim=True)

    return out, attackers


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, input):
        #         print(input.shape)
        '''
        input: batchsize* vector dimension * n 
        (1 by d by n)
        
        return 
            out : size =vector dimension, will be flattened afterwards
        '''
        out, attackers = k_means(input)

        return out, attackers