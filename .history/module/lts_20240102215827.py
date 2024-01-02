import torch
from module.data import MyDataset
import numpy as np
from module.utils import *


# 2. obtaining_LTS / selecting_LTS
def lts(model, X_train, y_train, lr_goal, n_not_fit, out_file, dtype = torch.float64):
    """
    X_train, y_trian, lr_goal, n_not_fit
    ---
    # output: train_loader, indices_lts, n
    train_loader: with lts
    n of train size 
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # predict and residuals
    y_pred = model(X_train)
    y_train = y_train.reshape(-1 ,1)
    resid_square = torch.square(y_pred - y_train).reshape(-1)

    # obtaining
    # prompt: find the indices of tensor < k only shape 1 tensor
    resid_square, sorted_indices = torch.sort(resid_square) # default ascending
    indices_lts = sorted_indices[resid_square < lr_goal**2]
    X_train_lts, y_train_lts = X_train[indices_lts], y_train[indices_lts]

    # check if obtaining is true. 0 is correct
    out_file.write(f"Total obtaining n: {len(indices_lts)}\n")
    out_file.write(f"obtaining n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts) > lr_goal**2).sum()}\n")
    print(f"Total obtaining n: {len(indices_lts)}\n")
    print(f"obtaining n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts) > lr_goal**2).sum()}\n")

    # selecting
    n = len(indices_lts) + n_not_fit
    indices_lts = sorted_indices[:n]
    X_train_lts, y_train_lts = X_train[indices_lts], y_train[indices_lts]

    # check if the selected is true. 1 is correct
    out_file.write(f"Total select n: {len(indices_lts)}\n")
    out_file.write(f"select n over lr goal: {(torch.square(model(X_train_lts) - y_train_lts)>lr_goal**2).sum()}\n")

    
    # to tensor 
    X_train_lts = torch.tensor(np.array(X_train_lts), dtype=torch.float64)
    y_train_lts = torch.tensor(np.array(y_train_lts), dtype=torch.float64)
    batch_size = 30

    # train loader of lts
    train_loader = torch.utils.data.DataLoader(
        MyDataset(X_train_lts, y_train_lts), 
        batch_size = batch_size, 
        shuffle=False, 
        drop_last = False)
    

    return train_loader, indices_lts, X_train_lts, y_train_lts, len(X_train_lts)