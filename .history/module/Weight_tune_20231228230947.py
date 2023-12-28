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

def validate(model, iterator, criterion):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for _, (X, y) in enumerate(iterator):
            preds = model(X)
            loss = criterion(preds, y)
            val_loss += loss.item()
    return preds, val_loss 

def eps_for_each(train_loader, model):
    eps = torch.zeros((1, 1), dtype=torch.float32)
    y_pred = torch.zeros((1, 1), dtype=torch.float32)
    for _, (X, y) in enumerate(train_loader):
        y = y.reshape(-1, 1)
        preds = model(X)
        eps = torch.cat([eps, (y-preds)**2], axis = 0)
        y_pred = torch.cat([y_pred, preds], axis = 0)
    eps, y_pred = eps[1:], y_pred[1:]
    return eps, y_pred

def module_weight_EU_LG_UA(model, train_loader, test_loader, 
                        epsilon, 
                        lr_lowerbound, 
                        optimizer, 
                        criterion, 
                        epochs,
                        data_name = None):
    """
    # need to save model after this module
    data_name: none = eth, copper = copper, ...
    """
    temp_save_path = "_temp/wt.pth"
    model.train()
    loss_old = 5e9
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(epochs):
        gc.collect()
        print(f"--------- module_EU_LG Epoch {epoch} ---------")

        train_loss = 0

        # forward operation
        for _, (X, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_loss_list.append(train_loss)
        print("train_loss:", train_loss)

        preds, test_loss = validate(model, test_loader, criterion)
        test_loss_list.append(test_loss)

        
        # stopping criteria 1
        eps, y_pred = eps_for_each(train_loader, model)

        print(max(eps))
        if max(eps) < epsilon:
            print(f"acceptable module at max eps {max(eps)}")
            acceptable = True
            acceptable_path = "acceptable/wt.pth"
            torch.save(model, acceptable_path)
            return acceptable, model, train_loss_list, test_loss_list
        
        # adjust lr
        if train_loss <= loss_old:
            print("Save model and lr increase")
            optimizer.param_groups[0]["lr"] *= 1.2
            torch.save(model.state_dict(), temp_save_path)
            loss_old = train_loss
        else:
            if optimizer.param_groups[0]['lr'] < lr_lowerbound:
                acceptable = False
                print(f"non acceptable module at max eps {max(eps)}")
                acceptable_path = "unacceptable/wt.pth"
                torch.save(model, acceptable_path)
                return acceptable, model, train_loss_list, test_loss_list            
            else:
                print("Restore model and lr decrease")
                state_dict = torch.load(temp_save_path)
                model.load_state_dict(state_dict)
                optimizer.param_groups[0]["lr"] *= 0.8
    
    # stopping criteria
    acceptable = False
    acceptable_path = "unacceptable/wt.pth"
    torch.save(model, acceptable_path)

    return acceptable, model, train_loss_list, test_loss_list