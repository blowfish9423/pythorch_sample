#https://www.youtube.com/watch?v=7q7E91pHoW4&t=13s

import torch
import torch.nn as nn
import numpy as np

''' #numpy实现
def cross_entropy(actual,predicted):
    loss = -np.sum(actual * np.log(predicted))
    return loss

# y must be one hot encoded
# if class 0: [1 0 0]
# if class 1: [0 1 0]
# if class 2: [0 0 1]

Y = np.array([1,0,0])

#y_pred has probabilities
Y_pred_good = np.array([0.7,0.2,0.1])
Y_pred_bad = np.array([0.1,0.3,0.6])

l1 = cross_entropy(Y,Y_pred_good)
l2 = cross_entropy(Y,Y_pred_bad)

print(f'Loss1 numpy:{l1:.4f}')
print(f'Loss2 numpy:{l2:.4f}')
'''

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
# nsamples * nclasses = 1x3
Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]])
Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]])

l1 = loss(Y_pred_good,Y)
l2 = loss(Y_pred_bad,Y)

print(l1.item())
print(l2.item())