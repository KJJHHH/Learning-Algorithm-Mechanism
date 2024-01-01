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
from utils import validate, check_acceptable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



def module_weight_EU_LG_UA(model, train_loader, test_loader, 
                        learning, 
                        lr_bound, 
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
    loss_old = 5e+9
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
        acceptable, eps, y_pred = check_acceptable(train_loader, model, epsilon)

        if acceptable:
            print(f"acceptable module max eps {max(eps)}")
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
                print("lr too small")
                print(f"non acceptable module at max eps {max(eps)}")
                torch.save(model, "unacceptable/wt.pth")
                return acceptable, model, train_loss_list, test_loss_list            
            else:
                print("Restore model and lr decrease")
                state_dict = torch.load(temp_save_path)
                model.load_state_dict(state_dict)
                optimizer.param_groups[0]["lr"] *= 0.8
    
    # stopping criteria
    torch.save(model, "unacceptable/wt.pth")

    return acceptable, model, train_loss_list, test_loss_list