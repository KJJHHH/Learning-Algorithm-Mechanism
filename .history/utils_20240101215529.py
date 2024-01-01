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