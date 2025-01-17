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

    for k in range(2, min(max_k, n - 1) + 1):
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

    max = -1*n

    b = []  # benign clients
    a = []  # attackers
    # Convert to a regular dictionary
    cluster_nodes = dict(cluster_nodes)
    print(cluster_nodes)
    # Print nodes in each cluster
    for cluster, nodes in cluster_nodes.items():
        print(f"Cluster {cluster}: Nodes {nodes}")
        for node in nodes:
            for node2 in nodes:
                if node != node2 and cost[node][node2] > max:
                    max = cost[node][node2]
    
    """# Classify points
    for i, size in enumerate(cluster_sizes):
        if size > min_cluster_size:  # Clusters larger than the smallest cluster are benign
            benign_indices.extend(np.where(labels == i)[0])
        else:  # Smallest clusters are considered attackers
            attackers.extend(np.where(labels == i)[0])

    # Aggregate benign points
    benign_indices = np.array(benign_indices)
    input_np = input.squeeze(0).numpy()  # Shape: (d, n)
    benign_points = input_np[:, benign_indices] if len(benign_indices) > 0 else np.zeros((d, 1))
    out = torch.tensor(np.mean(benign_points, axis=1, keepdims=True), dtype=torch.float32)  # Shape: (d, 1)"""

    #return out.unsqueeze(0), attackers, best_k  # Shape: (1, d, 1), list of attackers, optimal k


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