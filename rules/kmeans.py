import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances_argmin
import torch
import torch.nn as nn
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from collections import defaultdict

import numpy as np
from rules.correlations import C 
import matplotlib.pyplot as plt
from utils.logger import get_logger
from sklearn.decomposition import PCA

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

    # Convert input tensor to numpy for clustering
    X = input.squeeze(0).T.numpy()  # Shape: (n, d)

    # Try different values of k and calculate silhouette scores
    silhouette_scores = []
    cluster_results = {}

    for k in range(2,  n):
        kmeans = KMeans(n_clusters=k, n_init='auto', max_iter=max_iters, tol=tol, random_state=0)
        labels = kmeans.fit_predict(X)
        silhouette_scores.append(silhouette_score(X, labels))
        cluster_results[k] = (kmeans.cluster_centers_, labels)
        print("K: ", k, ", Silhouette Score: ", silhouette_score(X,labels))

    # Select the k with the best silhouette score
    best_k = np.argmax(silhouette_scores) + 2  # Adding 2 because range starts at k=2
    centroids, labels = cluster_results[best_k]

    # Identify benign clusters and attackers
    #cluster_sizes = np.array([np.sum(labels == i) for i in range(best_k)])
    #min_cluster_size = cluster_sizes.min()

    print("BEST K: ", best_k)

    visualize_clusters(X, labels, centroids)

    cost = C(input,n)

    cluster_nodes = defaultdict(list)

    # Assuming `labels` is the array of cluster labels for each point
    for idx, label in enumerate(labels):
        cluster_nodes[label].append(idx)

    # Convert to a regular dictionary
    cluster_nodes = dict(cluster_nodes)
    
    # To store the median scores of the correlation matrix for each cluster
    cluster_medians = {}

    # Print nodes in each cluster
    for cluster, nodes in cluster_nodes.items():
        print(f"Cluster {cluster}: Nodes {nodes}")
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
            median_value = None  # Handle cases with no pairwise values (e.g., single node clusters)
        
        cluster_medians[cluster] = median_value

    min = +1*n

    # Initialize empty lists to store benign clients, attackers, and temporary nodes
    benign_clients = []
    attackers = []
    temp_nodes = []

    # Print the median scores for each cluster
    for cluster, median in cluster_medians.items():
        # Print the median score for the current cluster
        print(f"Cluster {cluster}: Median Score {median}")
        
        # If the current cluster's median is smaller than the previously tracked minimum median
        if median < min:
            # Update the minimum median value
            min = median
            
            # If there were previous nodes in the temp_nodes list, add them to the attackers list
            if temp_nodes:
                attackers += temp_nodes
            
            # Add the current cluster's nodes to the temp_nodes list
            for node in cluster_nodes[cluster]:
                temp_nodes.append(node)
        else:
            # If the current cluster's median is not the smallest, consider it benign (non-attacker)
            # Add the current cluster's nodes directly to the attackers list
            for node in cluster_nodes[cluster]:
                attackers.append(node)
            
            # Reset temp_nodes as the current cluster does not have the smallest median
            temp_nodes = []
            

    print("Attackers: ", attackers)

    benign_clients.append([j for j in range(n) if j not in attackers])
    print("Benign clients:", benign_clients)

    out = torch.mean(input[:,[i for i in range(n) if i not in attackers]], dim=1, keepdim=True)

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