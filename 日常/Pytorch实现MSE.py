import torch
import numpy as np
from torch import nn

loss_fn = torch.nn.MSELoss(reduce=False,size_average=True)
#reduce = False，则size_average失效，直接返回向量形式的loss
#size_average = True,返回loss.mean();size_average = False,返回loss.sum()
#默认情况下，reduce = True,size_average = True
# a = np.array([[1,2],[3,4]])
# b = np.array([[2,3],[4,6]])
# input = torch.autograd.Variable(torch.from_numpy(a)) #将numpy数组(Tensor)转换为Variable
# target = torch.autograd.Variable(torch.from_numpy(b))
# loss = loss_fn(input.float(),target.float())
# print(input.float())
# print(target.float())
# print(loss)


# a = torch.ones(5,5,3)
# print(a)
# a = a.view(a.size(0),-1)
# print(a)
# loss = nn.CrossEntropyLoss()
# input = torch.randn(2,3,requires_grad=True)
# print(input)
# target = torch.empty(2,dtype=torch.long).random_(3)
# print(target)
# output = loss(input,target)
# print(output)
# output.backward()
# t = torch.tensor([[1,2],[3,4],[2,8]])
# print(torch.argmax(t,1))
# g = torch.tensor([[[1,2,3],[2,3,4],[5,6,7]], [[3,4,5],[7,6,5],[5,4,3]], [[8,9,0],[2,8,4],[7,5,3]]])
# print(g)
# print(torch.argmax(g,1))