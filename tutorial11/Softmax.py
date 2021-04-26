
#https://www.youtube.com/watch?v=7q7E91pHoW4&t=13s

#softmax and cross entropy

import torch
import torch.nn as nn
import numpy as np

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)

x = np.array([2.0,1.0,0.1])
outputs = softmax(x)
print('softmax numpy:',outputs)