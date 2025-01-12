import torch
import torch.nn as nn

import numpy as np
from rules.correlations import C 

import math
def density(r) :                   # calculates the average density of the correlation matrix (normalized sum of the correlations)
  sum = 0
  for i in range(len(r[0])) :
    sum += np.sum(r[i])
  return sum/(len(r[0]))

def fun(input) :                # input contains the clients weight updates
    n = input.shape[-1]         # number of clients
    r = C(input,n)              # correlation matrix
    d = 0
    h = []                      # temporary list of excluded nodes
    alt = []
    i = 0

    for k in range(n - 1) :                   # loop to exclude attacker nodes, one at a time, and recumputing the density after each elimination
        o = density(r)                        # original density
        temp = np.delete(r,i,0)               # remove row i from correlation matrix
        l = np.delete(temp,i,1)               # remove column i from correlation matrix
        d = density(l)                        # updated density
        if o < d :                            # if the desity increases by eliminating a node, that node is considered as an attacker and added to h
            r = l                             # update correlation matrix
            i-=1                              # correct index
            h.append(k)                       # append node to attacker vector
        i+=1

    input = input.squeeze(0)
    r2 = C(input,n)                           # calculate again correlation matrix to divide nodes into 2 subgroups
    alt = ([r2[i][j] for i in range(n) for j in range(n) if i in h and j in h and i != j])         # correlation between attacker nodes 
    m = ([r2[i][j] for i in range(n) for j in range(n) if i not in h and j not in h and i != j])   # correlation between benign nodes 
    #avg = np.sum(alt)/len(h)
    avg = np.median(alt)           # excluded nodes median correlation
    den = np.median(m)             # benign node's median correlation
    
    if len(h) < 2:      # if the number of attackers is too small (<2), all nodes are considered as attackers and excluded in the weight update calculation
        out = torch.mean(input[:,[i for i in range(n) if i not in h]], dim=1, keepdim=True)
        print(h)
        return out,h
    elif len(h) > n-2:    # # if the number of attackers is too high (>n-2), all nodes that are not in h are considered as attackers and excluded in the weight update calculation
        out = torch.mean(input[:,[i for i in range(n) if i in h]], dim=1, keepdim=True)     # computes aggregated weight update of benign nodes (the ones that are in h)
        print([i for i in range(n) if i not in h])         # attacker nodes are the ones that are not in h
        return out,[i for i in range(n) if i not in h]
    elif den < avg :                    # if the median correlation of not excluded nodes is < of the median correlation of excluded nodes, excluded nodes are attackers
        out = torch.mean(input[:,[i for i in range(n) if i not in h]], dim=1, keepdim=True)      # aggregated weight update considering only nodes not in h
        print(h)       # attackers
        print(avg)
        print(den)
        return out,h
    elif den > avg :         # viceversa, excluded nodes (the ones in h) are benign nodes, the other ones are attackers 
        out = torch.mean(input[:,[i for i in range(n) if i in h]], dim=1, keepdim=True)
        print([i for i in range(n) if i not in h])        # attackers (not in h)
        print(den)
        print(avg)
        return out,[i for i in range(n) if i not in h]

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
        out,attackers = fun(input)

        return out,attackers