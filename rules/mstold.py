import torch
import torch.nn as nn
import networkx as nx

import numpy as np
from rules.correlations import C 
import matplotlib.pyplot as plt
from utils.logger import get_logger

logger = get_logger()

def find(i,parent):           # searches for the parent of the group too which the node i belongs to
	while parent[i] != i:
		i = parent[i]
	return i
	
def union(i, j,parent):        # connects 2 subgraphs by letting one parent become parent also of the other group
	a = find(i,parent)
	b = find(j,parent)
	parent[a] = b
	return parent
	
def fun(input):                           # input contains the clients weight updates
    n = input.shape[-1]                   # number of clients
    parent = [i for i in range(n)]        # array to put connected nodes
    maxcost = 0                    
    INF = float('inf')                    # INF = +infinite (+oo)
    G = nx.DiGraph()          # graph
    G.add_nodes_from([i for i in range(n)])   # add n nodes to the graph
    cost = C(input,n)                         # correlation matrix

    edge_count = 0       # counter for the edges 
    while edge_count < n - 2:               # the loop searches for the non connected nodes with higher correlation values
        max = -1* INF                       # initialization of max to -infinite (-oo)
        a = -1
        b = -1
        for i in range(n):
            for j in range(n):
                if find(i,parent) != find(j,parent) and cost[i][j] > max:       # if nodes i and j are not in the same group, and with correlation higher than max,...
                    max = cost[i][j]
                    a = i
                    b = j
        parent = union(a, b, parent)         # connects the nodes 
        G.add_edge(a,b)                      # adds the connection to the graph by adding an edge
        edge_count += 1
        maxcost += max

    UG = G.to_undirected()         # converts the graph to undirected graph (no "arrows" i.e. no directions)
    sub_graphs = [UG.subgraph(c) for c in nx.connected_components(UG)]        # find sub-graphs of nodes
    min_d = -1*n
    p = []             # vector containing attackers
    k = [[],[]]        # vector containing valid nodes for each subgraph
    for i, sg in enumerate(sub_graphs) :              # for each subgraph, calculates the median weights
        k = [int(j) for j in sg.nodes]                # vector of integers representing nodes in the subgraphs
        f = [j for j in sg.edges.data("weight")]      # weights of the edges in the subgraph
        #print(f)
        #print([cost[x[0]] for x in f])
        if len(k) < 2 :                               # searches for the subgraph with maximum median weight to select valid nodes, nodes that are not included are considered attackers (vector p)
            p = k                                     # if there are less than 2 nodes in the subgraph, it is an attacker
            break
        if len(k) > n-2 :                             # if more than n-2, all nodes in the subgraph are benign, the others are attackers
            p = [j for j in range(n) if j not in k]
            break
        #d = np.average([np.sum(cost[x[0]]) for x in f])
        d = np.median([cost[x[0]][x[1]] for x in f])       # median of the weights in the subgraph
        print(f"Iter {i}, d: {d}")
        print(f"Iter {i}, k: {k}")
        if d > min_d :                  # if the median is higher then the former one, update it and save the nodes as attackers
            min_d = d
            p = k
    input = input.squeeze(0)        
    out = torch.mean(input[:,[i for i in range(n) if i not in p]], dim=1, keepdim=True)        # calculates the aggregated weight update, considering only benign clients
    print(f"Iter {i}, attackers: {p}")             # questi print dove sono? Non riesco a trovarli quando eseguo
    return out,p


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
        out, attackers = fun(input)

        return out, attackers