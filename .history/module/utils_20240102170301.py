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

def check_acceptable(train_loader, model, lr_goal):
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
    eps = torch.zeros((1, 1), dtype=torch.float32)
    y_pred = torch.zeros((1, 1), dtype=torch.float32)
    with torch.no_grad():
        for _, (X, y) in enumerate(train_loader):
            y = y
            preds = model(X)
            eps = torch.cat([eps, abs(y-preds)], axis = 0)
            y_pred = torch.cat([y_pred, preds], axis = 0)
    eps, y_pred = eps[1:], y_pred[1:]
    eps_ = abs()

    if max(eps) < lr_goal:
        return True, eps, y_pred
    else:
        return False, eps, y_pred