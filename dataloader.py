#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 1/5/2020 4:49 PM
@Author  : Yinyan Liu, The University of Sydney
@Email   : yinyan.yana.liu@gmail.com
"""
import numpy as np
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, list_dir, cv):
        '''
        Args: Read data
            list_dir: the path of sample list file
        '''
        super(Dataset, self).__init__()
        samples = []
        labels = []
        data = np.loadtxt(list_dir, delimiter=" ")
        if cv == 0:
            repeat_num = 1000
        elif cv == 1:
            repeat_num = 1000
        else:
            repeat_num = 1
        for i in range(0, repeat_num):
            for j in range(0, 2):
                x = data[:, j]
                if j == 0:
                    y = int(0)
                else:
                    y = int(1)
                # y = x[-1]
                x_norm = x/10
                samples.append(x_norm)
                labels.append(y)

        self.samples = samples
        self.labels = labels

    def __getitem__(self, index):
        x = self.samples[index]
        y = self.labels[index]
        return x, y

    def __len__(self):
        return len(self.samples)

if __name__ == "__main__":
    print()
    # you can test the dataset here