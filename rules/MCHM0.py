import torch
import torch.nn as nn

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from utils import utils


def fun(input) :
    
    scaler = MinMaxScaler()
    print(np.shape(input))
    arr = scaler.fit_transform(np.array(input.squeeze(0)))
    print(np.shape(arr))
    epsilon = 0.005
    
    for i in range(3) :
    
        for index,ele in enumerate(arr) :
    
            rmse = np.sqrt(np.array([np.sum([(i-j)**2 for j in ele]) for i in ele])/(len(ele)-1))

            avg = np.mean(ele)
    
            arr[index] = ele*(1.0 - rmse*np.sign(ele-avg)*0.1) + epsilon*int(i==0)
    
    out = torch.tensor(scaler.inverse_transform(arr), dtype = torch.float64).view(1,-1,len(ele))
    print(np.shape(out))
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