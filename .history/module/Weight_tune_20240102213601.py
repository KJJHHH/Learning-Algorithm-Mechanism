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
from module.utils import validate_loss, check_acceptable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def module_weight_EU_LG_UA(model, train_loader, test_loader, out_file,
                        lr_rate,
                        lr_bound, 
                        lr_goal, 
                        criterion, 
                        epochs,
                        data_name = None):
    """
    # need to save model after this module
    data_name: none = eth, copper = copper, ...
    # input
    model, train_loader, test_loader, **config_w
    ---
    # output
    acceptable, model, train_loss_list, test_loss_list
    """
    temp_save_path = "_temp/wt.pth"
    acceptable_path = "acceptable/wt.pth"
    unacceptable_path = "unacceptable/wt.pth"
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr_rate)
    loss_old = 5e+9
    train_loss_list = []
    test_loss_list = []
    
    for epoch in range(epochs):
        gc.collect()

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
        # out_file.write("train_loss:", train_loss, end = "\r")

        test_loss = validate_loss(model, test_loader, criterion)
        test_loss_list.append(test_loss)

        
        # stopping criteria 1
        acceptable, eps, y_pred = check_acceptable(train_loader, model, lr_goal)

        if acceptable:
            out_file.write(f"acceptable module max eps {max(eps)}")
            torch.save(model, acceptable_path)
            return acceptable, model, train_loss_list, test_loss_list
        
        # adjust lr
        if train_loss <= loss_old:
            # out_file.write("Save model and lr increase", end = "\r")
            optimizer.param_groups[0]["lr"] *= 1.2
            torch.save(model, temp_save_path)
            loss_old = train_loss
        else:
            if optimizer.param_groups[0]['lr'] < lr_bound:
                out_file.write("lr too small")
                out_file.write(f"non acceptable module at max eps {max(eps)}")
                torch.save(model, unacceptable_path)
                return acceptable, model, train_loss_list, test_loss_list            
            else:
                out_file.write("Restore model and lr decrease", end = "\r")
                model = torch.load(temp_save_path)
                optimizer.param_groups[0]["lr"] *= 0.8
    
    # stopping criteria
    out_file.write(f"finish epoch. non acceptable module at max eps {max(eps)}")    
    torch.save(model, unacceptable_path)

    return acceptable, model, train_loss_list, test_loss_list