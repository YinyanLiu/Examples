# -*- coding: utf-8 -*-
"""
Created on 2020/2/20 4:27 PM

@author: Yinyan Liu, The University of Sydney
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

def param(nnet, Mb=True):
    """
    Return number parameters(not bytes) in nnet
    """
    neles = sum([param.nelement() for param in nnet.parameters()])
    return neles / 10**6 if Mb else neles

class Model(nn.Module):
    def __init__(self,
                 kernel_size = 3,
                 stride = 1,
                 dropout=0.1):
        super(Model, self).__init__()

        self.kernel_size = kernel_size
        self.stride = stride
        self.dropout = dropout

        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=256, kernel_size=kernel_size, stride=stride, padding=1)
        self.conv3 = nn.Conv1d(in_channels=256, out_channels=16, kernel_size=kernel_size, stride=stride, padding=1)
        self.linear1 = nn.Linear(in_features=16*600, out_features=2048)
        self.linear2 = nn.Linear(in_features=2048, out_features=512)
        self.linear3 = nn.Linear(in_features=512, out_features=2)
        self.dp = nn.Dropout(dropout)
    def forward(self, x):
        if x.dim() == 2:
            x = th.unsqueeze(x, 1)
        # n x 1 x S => n x N x T
        w = F.relu(self.conv1(x)) # [batch, channel, T]

        w = F.relu(self.conv2(w))

        w = F.relu(self.conv3(w))

        w = self.dp(w)

        rw = w.view(w.size()[0], -1)
        # linear
        l = F.relu(self.linear1(rw))

        l = F.relu(self.linear2(l))

        out = F.relu(self.linear3(l))
        out = th.squeeze(out)
        return out
    @classmethod
    def load_model(cls, path):
        # Load to CPU
        package = th.load(path, map_location=lambda storage, loc: storage)
        model = cls.load_model_from_package(package)
        return model
    @classmethod
    def load_model_from_package(cls, package):
        model = cls(package['kernel_size'], package['stride'], package['non_linear'])
        model.load_state_dict(package['state_dict'])
        return model

    @staticmethod
    def serialize(model, optimizer, epoch, tr_loss=None, cv_loss=None):
        package = {
            # hyper-parameter
            'kernel_size': model.kernel_size, 'stride': model.stride, 'dropout':model.dropout,

            # state
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'epoch': epoch
        }
        if tr_loss is not None:
            package['tr_loss'] = tr_loss
            package['cv_loss'] = cv_loss
        return package


if __name__ == "__main__":
    # x = th.rand(4, 600)
    # nnet = Model()
    # x = nnet(x)
    # print(x.size())
    print()