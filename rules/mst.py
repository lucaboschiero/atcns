import torch
import torch.nn as nn
import networkx as nx

import numpy as np
from rules.correlations import C 

def find(i,parent):
	while parent[i] != i:
		i = parent[i]
	return i
	
def union(i, j,parent):
	a = find(i,parent)
	b = find(j,parent)
	parent[a] = b
	return parent
	
def fun(input):                           # input contains the clients weight updates
    n = input.shape[-1]                   # number of clients
    parent = [i for i in range(n)]        # array to put connected nodes
    maxcost = 0                    
    INF = float('inf')
    G = nx.DiGraph();          # graph
    G.add_nodes_from([i for i in range(n)])   # add n nodes to the graph
    cost = C(input,n)                         # correlation matrix

    edge_count = 0       # counter for the while loop 
    while edge_count < n - 2:               # the loop searches for the non connected nodes with higher correlation values
        max = -1* INF
        a = -1
        b = -1
        for i in range(n):
            for j in range(n):
                if find(i,parent) != find(j,parent) and cost[i][j] > max:
                    max = cost[i][j]
                    a = i
                    b = j
        parent = union(a, b, parent)
        G.add_edge(a,b)
        edge_count += 1
        maxcost += max

    UG = G.to_undirected()         # converts the graph to undirected graph (no "arrows" i.e. no directions)
    sub_graphs = [UG.subgraph(c) for c in nx.connected_components(UG)]        # find sub-graphs of nodes
    min_d = -1*n
    p = []             # vector containing attackers
    k = [[],[]]        # vector containing valid nodes for each subgraph
    for i, sg in enumerate(sub_graphs) :              # for each subgraph, calculates the median weights
        k = [int(j) for j in sg.nodes]
        f = [j for j in sg.edges.data("weight")]
        #print(f)
        #print([cost[x[0]] for x in f])
        if len(k) < 2 :                               # searches for the subgraph with maximum median weight to select valid nodes
            p = k                                     # nodes that are not included are considered attackers (vector p)
            break
        if len(k) > n-2 :
            p = [j for j in range(n) if j not in k]
            break
        #d = np.average([np.sum(cost[x[0]]) for x in f])
        d = np.median([cost[x[0]][x[1]] for x in f])
        print(d)
        print(k)
        if d > min_d :
            min_d = d
            p = k
    input = input.squeeze(0)        
    out = torch.mean(input[:,[i for i in range(n) if i not in p]], dim=1, keepdim=True)        # calculates the aggregated weight update, considering only benign clients
    print(p)             # questi print dove sono? Non riesco a trovarli quando eseguo
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