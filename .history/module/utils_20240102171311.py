from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
import torch, copy
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import  matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import gc

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def validate_loss(model, iterator, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(iterator):
            preds = model(X)
            loss = criterion(preds, y)
            val_loss += loss.item()
    return val_loss 

def check_acceptable(train_loader, model, lr_goal, X_train, y_train):
    """
    train_loader: train_loader
    model: model
    eps_bound: learning goal
    ---
    output: acceptable, eps, y_pred
    max eps
    acceptable
    y_pred 
    """
    eps_square = torch.zeros((1, 1), dtype=torch.float32)
    y_pred = torch.zeros((1, 1), dtype=torch.float32)
    with torch.no_grad():
        for _, (X, y) in enumerate(train_loader):
            y = y
            preds = model(X)
            eps_square = torch.cat([eps_square, torch.square(y-preds)], axis = 0)
            y_pred = torch.cat([y_pred, preds], axis = 0)
    eps_square, y_pred = eps_square[1:], y_pred[1:]



    eps_ = torch.square(y_train - model(X_train))
    print("===================================")
    for i  in eps_:
        if i not in eps_square:
            print(123123123123)
    print("===================================")



    if max(eps_square) < lr_goal**2:
        return True, eps_square, y_pred
    else:
        return False, eps_square, y_pred