import numpy as np

def accuracy(y_true, y_pred):
    ltrue = len(y_true)
    lpred = len(y_pred)
    if ltrue != lpred or ltrue < 0:
        raise Exception("invalid size of y_true or y_pred")
    return (y_true == y_pred).astype(int).mean()

def r2(y_true, y_pred):
    y_bar = y_true.mean()
    ss_tot = ((y_true-y_bar)**2).sum()
    ss_res = ((y_true-y_pred)**2).sum()
    return 1 - (ss_res/ss_tot)
