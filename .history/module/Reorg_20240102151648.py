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
import gc, os
from module.Weight_tune import *
from module.model import TwoLayerNet
from module.utils import validate, check_acceptable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class reorganising():
    def __init__(
            self, previous_module, train_loader, test_loader, 
            criterion = nn.MSELoss(), 
            epochs = 50, 
            lr_rate = .001, 
            lr_bound = .00001,  
            lr_goal = .0001, 
            print_reg = False,
            print_w_tune = False,
            validate_run = False):
        super(reorganising, self).__init__()
        # Reorganising module
        # regular module
        #   model => self.model: main model (final model) in regualrising module. 
        # wieght tune module
        #   model => create new model when process  
        """
        previous_module: "wt" or "Cram"
        hidden_dim: hidden dimension of model
        epochs: number of epochs
        criterion: loss function
        lr_reg: learning rate for regularization
        lr_w_tune: learning rate for weight tuning
        lr_bound_reg: learning rate lower bound for regularization
        lr_bound_w_tune: learning rate lower bound for weight tuning      
        eps_reg: early stopping epsilon for regularization
        eps_w_tune: early stopping epsilon for weight tuning
        validate_run: use model with test loader and check loss if True
        """        

        # Initialise: check if accetable SLFN (wt.pth) exist  
        self.acceptable_path = f'acceptable/{previous_module}.pth'
        if os.path.exists(self.acceptable_path):
            print(f"Acceptable SLFN exist in '{self.acceptable_path}'.")
            self.model = torch.load(self.acceptable_path)
        else:
            print(f"Acceptable SLFN not exist in '{self.acceptable_path}'.")
            self.model = None        
                         
        self.input_dim = self.model.layer_1.weight.data.shape[1]
        self.hidden_dim = self.model.layer_1.weight.data.shape[0]
        self.train_loader = train_loader
        self.test_loader = test_loader

        self.epochs = epochs
        self.criterion = criterion
        self.lr_rate = lr_rate

        self.lr_bound = lr_bound
        self.lr_goal = lr_goal

        self.print_reg = print_reg
        self.print_w_tune = print_w_tune
        self.validate_run = validate_run
    
    def validate(self, model, iterator):
        """
        model: model for evaluate
        iterator: loader
        criterion: loss function        
        """
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for _, (X, y) in enumerate(iterator):
                preds = model(X)
                loss = self.criterion(preds, y)
                val_loss += loss.item()
        val_loss /= len(iterator)
        return preds, val_loss 

    """
    need model, optimizer
    """
    def module_weight_EU_LG_UA(self, model):
        """
        # do not have to save model after this module
        model: trimm model
        optimizier: optimizer for model
        """

        temp_save_path = "_temp/model.pth"
        model.train()
        optimizer = optim.Adam(model.parameters(), lr = self.lr_rate)
        loss_old = 5e+7
        train_loss_list = []
        test_loss_list = []
        
        for epoch in range(self.epochs):
            gc.collect()
            print(f"|{epoch/self.epochs*100: .2f}%| Epoch {epoch}", end="\r")

            train_loss = 0

            # forward operation
            for _, (X, y) in enumerate(self.train_loader):
                
                optimizer.zero_grad()
                preds = model(X)
                loss = self.criterion(preds, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(self.train_loader)
            train_loss_list.append(train_loss)
            if self.print_w_tune:
                print("train_loss:", train_loss)

            if self.validate_run:
                preds, test_loss = self.validate(model, self.test_loader)
                test_loss_list.append(test_loss)

            
            # stopping criteria 1
            acceptable, eps, y_pred = check_acceptable(self.train_loader, model, self.lr_goal)

            if self.print_w_tune:
                print(f"max eps sqaure: {max(eps)}")
            if acceptable:
                print(f"acceptable module at max eps {max(eps)}")
                return acceptable, model, train_loss_list, test_loss_list
            
            # adjust lr
            if train_loss <= loss_old:
                if self.print_w_tune:
                    print("Save model and lr increase")
                optimizer.param_groups[0]["lr"] *= 1.2
                torch.save(model.state_dict(), temp_save_path)
                loss_old = train_loss
            else:
                if optimizer.param_groups[0]['lr'] < self.lr_bound:
                    print(f"non acceptable module at max eps {max(eps)}")
                    return acceptable, model, train_loss_list, test_loss_list            
                else:
                    if self.print_w_tune:
                        print("Restore model and lr decrease")
                    state_dict = torch.load(temp_save_path)
                    model.load_state_dict(state_dict)
                    optimizer.param_groups[0]["lr"] *= 0.8
        

        return acceptable, model, train_loss_list, test_loss_list
    
    """
    model: self.model, optimizer: define in function
    """
    def regularising_EU_LG_UA(self):  
        """
        No need to send:
        model: self.model
        optimizer: optimizer for self.model
        """
        # define regularising
        # L7 p 47
        # Classmate p 13?

        self.model.train()     # Enter Train Mode
        temp_save_path = "_temp/reg.pth"
        torch.save(self.model, temp_save_path)
        train_loss_list = [] 
        test_loss_list = []
        loss_old = self.validate(self.model, self.train_loader)
        acceptable, eps, y_pred = check_acceptable(self.train_loader, self.model, self.lr_goal)
        max_eps_old = max(eps)

        optimizer = torch.optim.Adam(self.model.parameters(), lr = self.lr_rate)

        for epoch in range(self.epochs):
            gc.collect()
            print(f"|{epoch/self.epochs*100: .2f}%| Epoch {epoch}", end="\r")

            train_loss = 0

            # forward operation
            for _, (X, y) in enumerate(self.train_loader):
                
                optimizer.zero_grad()
                preds = self.model(X)
                loss = self.criterion(preds, y) + self.reg(self.hidden_dim, X.shape[0])
                loss.backward()
                optimizer.step()
                train_loss += loss.item()


            # train loss             
            train_loss /= len(self.train_loader)
            train_loss_list.append(train_loss)
            if self.print_reg:
                print("train_loss:", train_loss)
                print(f"regular term: {self.reg(self.hidden_dim, X.shape[0])}")

            # Validate
            if self.validate_run:
                preds, test_loss = self.validate(self.model, self.test_loader)
                test_loss_list.append(test_loss)
            
            # stopping criteria 1
            acceptable, eps, y_pred = check_acceptable(self.train_loader, self.model, self.lr_goal)
            
            # adjust lr
            if train_loss <= loss_old:
                if self.print_reg:
                    print("Save model and lr increase")
                
                # stop 2: if max eps > lr goal 
                if max(eps) >= self.lr_goal:
                    # restore model and end reg module
                    self.model = torch.load(temp_save_path)
                    return train_loss_list, test_loss_list
                else:
                    optimizer.param_groups[0]["lr"] *= 1.2
                    torch.save(self.model, temp_save_path)
                    loss_old = train_loss
                    continue
                
            else:
                # stop 3. if lr too small
                # NOTE: also acceptable, since restore from previous acceptable model
                self.model = torch.load(temp_save_path)
                if optimizer.param_groups[0]['lr'] < self.lr_bound:
                    return train_loss_list, test_loss_list            
                else:
                    if self.print_reg:
                        print("Restore model and lr decrease")
                    self.model = torch.load(temp_save_path)
                    optimizer.param_groups[0]["lr"] *= 0.8

        return train_loss_list, test_loss_list  

    def reg(self, p, m):
        # reg term
        """
        p: first hidden node size
        m: input sample size (batch size)
        """
        reg_lambda = ( 0.001 / (p+1 + p*(m+1)))
        reg = 0

        names = []
        for name, tensor in self.model.named_parameters():
            # print(f"{name}")
            names.append(name)
        for i in names:
            weights = self.model.state_dict()[i]
            reg += weights.square().sum()
        return reg*reg_lambda

    def trim_model_nodes(self, k):
        # del nodes k
        """
        model: model to trimmed
        k: node to be trimmed
        """
        trim_model = TwoLayerNet(self.input_dim, self.hidden_dim-1, 1)
        param = self.model.state_dict()
        delete_status = []
        # Update nodes (del)
        for name in param:
            weights = param[name]

            # k start from 1, the k-1 row in weights will be delete
            if name == "layer_1.weight":
                trim_model.layer_1.weight.data  = torch.cat([weights[:k-1, :], weights[k:, :]], dim=0)
                if False in (trim_model.layer_1.weight.data == torch.cat([weights[:k-1, :], weights[k:, :]], dim=0)):
                    delete_status.append(False)
                else:
                    delete_status.append(True)
            elif name == "layer_1.bias":
                trim_model.layer_1.bias.data = torch.cat([weights[:k-1], weights[k:]], dim=0)
                if False in (trim_model.layer_1.bias.data == torch.cat([weights[:k-1], weights[k:]], dim=0)):
                    delete_status.append(False)
                else:
                    delete_status.append(True)

            elif name == "layer_2.weight":
                trim_model.layer_2.weight.data = torch.cat([weights[:, :k-1], weights[:, k:]], dim=1)
                if False in (trim_model.layer_2.weight.data == torch.cat([weights[:, :k-1], weights[:, k:]], dim=1)):
                    delete_status.append(False)
                else:
                    delete_status.append(True)
            elif name == "layer_2.bias":
                pass
                
        if False not in delete_status:
            print("Trim model: Copy model and delete nodes success")
        else:
            print("Trim model: Copy model and delete nodes error")
            
        return trim_model
    
    def reorganising(self):
        # 1. Check nodes: if delete the node, the model trained by weight tuning module 
        #   still can be accepted, delete the node
        # 2. Store the final model in: final_model/Reorg
        """
        self.model: model to be reorganised. main model.
        trim_model: model trimmed from regular module model. 
                    If acceptable, self.model = trim_model (no trained with weight tune).
        """
        # Initialise: check if accetable SLFN (wt.pth) exist        
        if self.model == None:
            return None 

        # the k node to be check
        print('=================== reorganising ===================')
        k = 1 
        check_n = 0
        while k<=self.hidden_dim:
            print(f"[{check_n/self.hidden_dim*100: .2f}%] ------------> Checking nodes...")
            print(self.model)            
            print(f"    --> Start regularising_EU_LG_UA")
            check_n += 1
            acceptable, train_loss_list, test_loss_list = \
                self.regularising_EU_LG_UA()
            
            
            # A new model trimmed from regualr module model
            # Use the trim model in weigt tuning
            trim_model = self.trim_model_nodes(k)
            
            # Training the trim model with weight tune
            # Return trained trim model
            print(f"    --> Start module_EU_LG")
            acceptable, model, train_loss_list, test_loss_list = \
                self.module_weight_EU_LG_UA(trim_model)
            
            if acceptable == True:
                print("!!!Trim node from the model, hidden nodes decrease by 1!!!")
                self.hidden_dim-=1
                self.model = trim_model
                if self.hidden_dim == 1:
                    print("mdoel only 1 hidden nodes")
                    return model

            else:
                k+=1
        
        # Store all model, not just state dict
        torch.save(self.model, 'acceptable/Reorg.pth')


    