import torch
import torch.nn as nn
import numpy as np

from rules.correlations import C 

def fun(input) :
    n = input.shape[-1]
    a = C(input,n)
    max = -1*n
    ind = -1
    att = []
    for i in range(n) :
        if np.sum(a[i]) > max :
            max = np.sum(a[i])
            ind = i
    for j in range(len(a[i])) :
        if a[i][j] < 0 or j in att:
            att.append(j)
            t = list(np.where(a[j] >= np.mean(a[i]))[0])
            for y in np.array(t) :
                if a[i][y] < np.mean(a[i]) and y not in att:
                    att.append(y)
    if len(set(att)) > n/2 :
        att = [i for i in range(n) if i not in set(att)]
    input = input.squeeze(0) 
    print(att)
    out = torch.mean(input[:,[i for i in range(n) if i not in att]], dim=1, keepdim=True)    
    return out

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
        out = fun(input)

        return out