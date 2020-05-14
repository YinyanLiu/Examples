#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
@Time    : 1/27/2020 11:23 AM
@Author  : Yinyan Liu, The University of Sydney
@Email   : yinyan.yana.liu@gmail.com
"""
import argparse
import torch
import os
from pathlib import Path
from dataloader import *
from net import Model
from torch.utils.data import DataLoader
from learning.metrics import get_mae, f1_score

parser = argparse.ArgumentParser('Evaluate separation performance using ')
parser.add_argument('--data_dir', type=str,
                    default=Path('dataset/tr_data.txt'),
                    help='directory including test data'),

def test(args):
    # Load data
    dataset = Dataset(list_dir=args.data_dir, cv=2)
    data_loader = DataLoader(dataset,
                             batch_size=1,
                             num_workers=0)
    model_file = Path('exp/final.pth.tar')
    # Load model
    model = Model.load_model(model_file)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')
    model = model.to(device=device)
    model.eval()
    F1_ALL = []
    MAE_ALL = []
    with torch.no_grad():
        for i, (data) in enumerate(data_loader):
            # Get batch data
            x, y = data
            x = x.to(device=device, dtype=torch.float)
            y = y.to(device=device, dtype=torch.float)
            est = model(x)
            y_hat = est[0, 0]
            print('-' * 60)
            mae = get_mae(target=y, prediction=y_hat)
            F1 = f1_score(target=y, prediction=y_hat, threshold=0)

            print('test %d' %i)
            print('mae:', mae)
            print('f1:', F1)
            F1_ALL.append(F1)
            MAE_ALL.append(mae)

if __name__ == '__main__':
    # args = parser.parse_args()
    # print(args)
    # test(args)
    loss = torch.nn.CrossEntropyLoss()
    input = torch.randn(1, 2, requires_grad=True)
    target = torch.empty(1, dtype=torch.long).random_(2)
    output = loss(input, target)
    print(input)
    print(target)
    print(output)