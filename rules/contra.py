import torch
import torch.nn as nn

'''
FoolsGold

retrieved from https://github.com/DistributedML/FoolsGold/blob/master/deep-fg/fg/foolsgold.py

Reference:
Fung, Clement, Chris JM Yoon, and Ivan Beschastnikh. "The Limitations of Federated Learning in Sybil Settings." 23rd International Symposium on Research in Attacks, Intrusions and Defenses ({RAID} 2020). 2020.

'''

import numpy as np
import sklearn.metrics.pairwise as smp


# Takes in grad
# Compute similarity
# Get weightings
def CONTRA(grads):
    n_clients = grads.shape[0]
    r = np.ones(n_clients)
    cs = smp.cosine_similarity(grads) - np.eye(n_clients)
    #maxcs = np.max(cs, axis=1)
    temp = cs[np.argsort(cs, axis = 1)]
    n = 20
    maxcs = np.average(temp[ : -n], axis = 1)[0]

    for i in range(n_clients) :
        if  maxcs[i] > 0.05 :
            r[i] = r[i] - 0.1
        else :
            r[i] = r[i] + 0.1
    
    r = r / np.max(r)
    
    n = 10
    
    ele = r[np.argsort(r)[-n]]
    
    C = [1 if i >= int(ele) else 0 for i in r]

    # pardoning
    for i in range(n_clients):
        for j in range(n_clients):
            if i == j:
                continue
            if maxcs[i] < maxcs[j]:
                cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
    wv = 1 - (np.max(cs, axis=1))
    wv[wv > 1] = 1
    wv[wv < 0] = 0

    # Rescale so that max value is wv
    wv = wv / np.max(wv)
    wv[(wv == 1)] = .99

    # Logit function
    wv = (np.log(wv / (1 - wv)) + 0.5)
    wv[(np.isinf(wv) + wv > 1)] = 1
    wv[(wv < 0)] = 0

    return wv*C


def adaptor(input):
    '''
    compute CONTRA 
    
    input : 1* vector dimension * n
    
    return 
        foolsGold :  vector dimension 
    '''
    x = input.squeeze(0)
    x = x.permute(1, 0)
    w = CONTRA(x)
    print(w)
    attackers = [i for i in list(range(x.shape[0])) if w[i] == 1]
    w = w / w.sum()
    out = torch.sum(x.permute(1, 0) * w, dim=1, keepdim=True)
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
        out, attackers= adaptor(input)
        print(type(out))

        return out, attackers
