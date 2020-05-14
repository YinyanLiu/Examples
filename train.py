#!/usr/bin/env python

import argparse
import torch
from pathlib import Path
from dataloader import *
from torch.utils.data import DataLoader
from learning.weight_init import weight_init
from net import Model

import warnings
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser("energy disaggreation ")
# Task related
parser.add_argument('--train_dir', type=str,
                    default=Path('dataset/tr_data.txt'),
                    help='directory including train data'),
parser.add_argument('--valid_dir', type=str,
                    default=Path('dataset/tr_data.txt'),
                    help='directory including validation data'),
parser.add_argument('--t_len', default=600, type=int,
                    help='Length of input data')

# Training config
parser.add_argument('--epochs', default=100, type=int,
                    help='Number of maximum epochs'),
parser.add_argument('--max_norm', default=5, type=float,
                    help='Gradient norm threshold to clip'),
parser.add_argument('--use_cuda', default=True,
                    help='if use cuda '),
parser.add_argument('--loss', default="MSE", type=str,
                    help='loss type: MSE or l1'),

# minibatch
parser.add_argument('--batch_size', default=4, type=int,
                    help='batch size'),
# optimizer
parser.add_argument('--optimizer', default='Adam', type=str,
                    choices=['RMSprop', 'Adam'],
                    help='Optimizer (support sgd and adam now)'),
parser.add_argument('--lr', default=1e-3, type=float,
                    help='Init learning rate'),

# save and load model
parser.add_argument('--save_folder', default=Path('exp/train'),
                    help='Location to save epoch models'),
parser.add_argument('--save_name', default='final.pth.tar',
                    help='best model name'),
# logging
parser.add_argument('--print_freq', default=10, type=int,
                    help='Frequency of printing training infomation'),
def main(args):
    #------------ start to prepare dataset ------------'
    tr_dataset = Dataset(list_dir=args.train_dir, cv=0)
    cv_dataset = Dataset(list_dir=args.valid_dir, cv=1)

    tr_loader = DataLoader(tr_dataset,
                           batch_size=args.batch_size,
                           shuffle=True,
                           num_workers=0)
    cv_loader = DataLoader(cv_dataset,
                           batch_size=2,
                           shuffle=False,
                           num_workers=0)
    #'------------------ model -----------------------'
    model = Model(kernel_size = 3,
                 stride = 1,
                 dropout=0.1)
    print(model)
    model.apply(weight_init)

    if args.use_cuda == True and torch.cuda.is_available():
        device = torch.device("cuda")
        model = torch.nn.DataParallel(model)
    else:
        device = torch.device('cpu')

    model = model.to(device=device)

    # optimizer
    if args.optimizer == 'RMSprop':
        optimizier = torch.optim.RMSprop(model.parameters(),
                                      lr=args.lr)
    elif args.optimizer == 'Adam':
        optimizier = torch.optim.Adam(model.parameters(),
                                      lr=args.lr)
    else:
        print("Not support optimizer")
        return RuntimeError('Unrecognized optimizer')

    # Loss
    # Loss = torch.nn.MSELoss()

    train_total_loss = []
    cv_total_loss = []
    best_loss = float("inf")
    no_improve_nums = 0
    # ---------------------------------- Training ------------------------
    for epoch in range(0, args.epochs):
        model.train()
        tr_loss = torch.tensor(0.0)
        for i, (data) in enumerate(tr_loader):
            x, y = data
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.long)
            est = model(x)
            loss = torch.nn.functional.cross_entropy(input=est, target=y)
            # loss = Loss(input=est, target=y)
            tr_loss += loss
            optimizier.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=5)
            optimizier.step()

        tr_loss = tr_loss / i
        train_total_loss.append(tr_loss.cpu().detach().numpy())
        print('-' * 80)
        print('Epoch %d End train with loss: %.3f' % (epoch, tr_loss))
        print('-' * 80)

        # ---------------------------- validation  ---------------------------
        model.eval()
        cv_loss = torch.tensor(0.0)
        with torch.no_grad():
            for j, (data) in enumerate(cv_loader):
                x, y = data
                x = x.to(device=device, dtype=torch.float)
                y = y.to(device=device, dtype=torch.long)

                est = model(x)
                loss = torch.nn.functional.cross_entropy(input=est, target=y)
                # loss = Loss(input=est, target=y)
                cv_loss += loss
                if j %5==0:
                    print(
                        'Epoch %d, Iter: %d,  Loss: %.3f' % (epoch, j, loss))
            cv_loss = cv_loss/ j
            cv_total_loss.append(cv_loss.cpu().detach().numpy())
            print('-' * 80)

            if best_loss > cv_loss:
                best_loss = cv_loss
                torch.save(model.module.serialize(model.module,
                                                  optimizier,
                                                  epoch + 1,
                                                  tr_loss=tr_loss,
                                                  cv_loss=cv_loss),
                           args.save_folder/args.save_name)
                print("Find best validation model, saving to %s" % str(args.save_folder/args.save_name))
                no_improve_nums = 0
            else:
                print('no improve ...')
                no_improve_nums += 1
                if no_improve_nums >= 3:
                    optim_state = optimizier.state_dict()
                    optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / 2.0
                    optimizier.load_state_dict(optim_state)
                    print('Reduce learning rate to lr: %.8f' % optim_state['param_groups'][0]['lr'])
                if no_improve_nums >= 6:
                    print('No improve for 6 epochs, stopping')
                    break
            print('Epoch %d End validation with loss: %.3f, best loss: %.3f' % (epoch, cv_loss, best_loss))
            print('-' * 80)
if __name__ == '__main__':
    # # generate samples
    # x_all = []
    # for i in range(0, 2):
    #     x = np.random.randint(0, 10, [600])
    #     x_all.append(x)
    # np.savetxt('dataset/tr_data.txt', np.array(x_all).T)
    args = parser.parse_args()
    print(args)
    main(args)


