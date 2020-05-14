# -*- coding: utf-8 -*-
"""
Created on 2020/3/11 7:11 PM

@author: Yinyan Liu, The University of Sydney
"""
import numpy as np
import matplotlib.pyplot as plt
NORM_VALUE = [80.0, 7600.0]
APPLIANCES = ['kettle', 'washing_machine', 'dishwasher', 'toaster', 'fridge']

def test_plt(mixture, y_pred, y_true, save_path, index):
    '''
    :param y_pred: [batch, time, appliances] such as [1, 600]
    :param y_true: [batch, time, appliances] such as [1, 600]
    :param save_path: save directory
    :param index: int
    :return:
    '''
    T, A = np.shape(y_pred)
    x = np.arange(1, T + 1)
    save_name = save_path / '{}.png'.format(index)
    mix = mixture*(NORM_VALUE[1]-NORM_VALUE[0]) + NORM_VALUE[0]

    plt.plot(x, mix, label='Aggregate load')
    for i in range(0, A):
        app_true = y_true[:, i]
        app_pred = y_pred[:, i]
        app_pred[app_pred <30] = 0
        plt.plot(x, app_true, label='{}_truth'.format(APPLIANCES[i]))
        plt.plot(x, app_pred, label='{}_est'.format(APPLIANCES[i]))

    plt.ylabel('Active power (W)')
    plt.xlabel('samples')
    plt.legend()
    plt.savefig(save_name)
    plt.close()

def get_mae(target, prediction):
    '''
    compute the  absolute_error
    Parameters:
    ----------------
    target: the groud truth , np.array
    prediction: the prediction, np.array
    '''
    assert (target.shape == prediction.shape)
    err_apps = []
    for i in range(np.shape(target)[-1]):
        err = np.mean(np.abs(target[:, i] - prediction[:, i]))
        err_apps.append(err)
    return err_apps

from sklearn.metrics import confusion_matrix
def f1_score(target, prediction, threshold):
    tar = np.array(target)
    tar[tar<=threshold] = 0
    tar[tar>threshold] = 1
    tar = np.array(tar, dtype=np.int)
    pred = np.array(prediction)
    pred[pred <= threshold] = 0
    pred[pred > threshold] = 1
    pred = np.array(pred, dtype=np.int)
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true=tar, y_pred=pred, labels=[0, 1])

    if tp + fp == 0 and tp == 0:
        precision = 1.0
    else:
        precision = (tp) / (tp + fp)
    if tp + fn == 0 and tp == 0:
        recall = 1.0
    else:
        recall = tp / (tp + fn)
    if recall + precision == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return f1