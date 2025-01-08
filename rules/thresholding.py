import torch
import torch.nn as nn

import numpy as np
from rules.correlations import C
import statistics as sts
import pandas as pd

def fun(input) :
  n = input.shape[-1]
  try :
      ni = pd.read_csv("ni.csv", header = None, sep = ',').values
  except :
      ni = [i for i in range(n)]
  a = C(input,n)
  m = np.mean([a[i] for i in ni])
  s = np.std([a[i] for i in ni])
  count_l = 0
  count_m = 0
  count_h = 0
  att = []
  b = a
  for i in range(n) :
    count_l = 0
    count_m = 0
    count_h = 0
    for j in range(n) :
      if i != j :
        if a[i][j] > m + 1.5*s :
          count_h += 1
        elif a[i][j] < m - 1.5*s :
          count_l += 1
        else :
          count_m += 1
    if count_l > count_h :
        att.append(i)
  #ni = [i for i in range(n) if i not in att]
  input = input.squeeze(0)        
  out = torch.mean(input[:,[i for i in range(n) if i not in att]], dim=1, keepdim=True)
  print(att)
  ni = [i for i in range(n) if i not in att]
  pd.DataFrame(ni).to_csv("ni.csv",header = None, index = None, sep = ',')
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