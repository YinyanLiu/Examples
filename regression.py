# -*- coding: utf-8 -*-
"""
Created on 2020/5/14 4:42 PM

@author: Yinyan Liu, The University of Sydney
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(1, 10, 100), dim=1)  # 这句话是把x本来是个一维的数据进行升维
y = x.pow(2) + 0.2 * torch.rand(x.size())  # 把 x^2乘2的数据加上一个随机噪声。 rand应该是生成0-1之间的一个均匀分布

plt.scatter( x.data.numpy(),y.data.numpy() )#打印散点图
plt.show()

class Net(torch.nn.Module):  # 继承torch.nn.Moudle这个模块，可以调用这个模块的接口
    def __init__(self, n_features, n_hidden, n_output):  # 搭建计算图的一些变量的初始化
        super(Net, self).__init__()  # 将self（即自身）转换成Net类对应的父类，调用父类的__init__（）来初始化
        self.hidden = torch.nn.Linear(n_features, n_hidden)
        self.predict = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):  # 前向传播，搭建计算图的过程
        x = F.relu(self.hidden(x))
        x = self.predict(x)
        return x

def mre(y_true, y_pred):
    with torch.no_grad():
        err = torch.mean(torch.abs((y_true - y_pred) / (y_true)))
    return err

net = Net(1, 100, 1)

print(net)

plt.ion()  # matplotlib变成一个实时打印的状态
plt.show()

optimizer = torch.optim.Adam(net.parameters(), lr=0.05)  # 随机梯度下降算法
loss_func = torch.nn.MSELoss()  # 代价函数,应该是二次代价函数。

for t in range(1000):
    prediction = net(x)
    loss = loss_func(prediction, y)
    optimizer.zero_grad()  # 清空上一部的梯度值，否则应该会累加上去。
    loss.backward()  # 反向传播就散梯度值
    optimizer.step()  # 用计算得到的梯度值，来更新神经网络里面的各个参数

    if t % 5 == 0:
        plt.cla()  # 清楚上幅图的散点
        plt.scatter(x.data.numpy(), y.data.numpy())  # 原始数据的散点图
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'LOSS=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
        plt.pause(0.1)
        print('mre err:', mre(y_pred=prediction, y_true=y))
plt.ioff()
plt.show()


