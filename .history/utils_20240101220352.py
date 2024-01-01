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
from module.model import TwoLayerNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def validate(model, iterator, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(iterator):
            preds = model(X)
            loss = criterion(preds, y)
            val_loss += loss.item()
    return preds, val_loss 

def check_acceptable(train_loader, model, eps_bound):
    """
    train_loader: train_loader
    model: model
    eps_bound: learning goal
    ---
    output:
    max eps
    acceptable
    y_pred 
    """
    eps = torch.zeros((1, 1), dtype=torch.float32)
    y_pred = torch.zeros((1, 1), dtype=torch.float32)
    with torch.no_grad():
        for _, (X, y) in enumerate(train_loader):
            y = y.reshape(-1, 1)
            preds = model(X)
            eps = torch.cat([eps, abs(y-preds)], axis = 0)
            y_pred = torch.cat([y_pred, preds], axis = 0)
    eps, y_pred = eps[1:], y_pred[1:]

    if max(eps)  eps_bound:
        return False, max(eps), y_pred
    else:
        return True, max(eps), y_pred