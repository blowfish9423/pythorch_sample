#https://zhuanlan.zhihu.com/p/116014061

import torch
'''
A
    tensor([[1., 1., 1.],
        [1., 1., 1.]])
'''
A=torch.ones(2,3)    #2x3的张量（矩阵）
print("A:\n",A,"\nA.shape:\n",A.shape,"\n")
'''
B:
    tensor([[2., 2., 2.],
        [2., 2., 2.]])
'''
B=2*torch.ones(2,3)  #4x3的张量（矩阵）
print("B:\n",B,"\nB.shape:\n",B.shape,"\n")
'''
C:
 tensor([[1., 1., 1.],
        [1., 1., 1.],
        [2., 2., 2.],
        [2., 2., 2.]])
C.shape:
 torch.Size([4, 3])
'''
C=torch.cat((A,B),0)  #按维数0（行）拼接 维数1（列）拼接
print("C:\n",C,"\nC.shape:\n",C.shape,"\n")
'''
D:
 tensor([[1., 1., 1., 2., 2., 2.],
        [1., 1., 1., 2., 2., 2.]])
torch.Size([2, 6])
'''
D = torch.cat([A,B],1)
print("D:\n",D,"\nC.shape:\n",D.shape,"\n")