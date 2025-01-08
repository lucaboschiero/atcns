import torch
import torch.nn as nn

import numpy as np
from utils import utils
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


def fun(value) :
    pca = PCA(n_components='mle')
    output = pca.fit_transform(value.numpy())   
    out = torch.tensor(output, dtype = torch.float64)
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
        out = fun(input.squeeze(0))
        print("Number of Dimensions kept are",out.shape[1])
        output = torch.sum(out, dim = 1) / out.shape[1]

        return output