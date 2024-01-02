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
from module.model import TwoLayerNet
from module.utils import validate, check_acceptable

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class cramming(nn.Module):
    def __init__(self, model, train_loader, X_train, y_train, lr_goal, s):
        
        """
        model: 
        train_loader
        X_train:
        y_train:
        lr_goal: maximum epsilon accepted for EACH sample
                   as epsilon in weight tuning module and eps_reg, eps_w_tune in module above
        s: float (tiny) that r*(Xc - Xk) != 0 and (s - r*(Xc - Xk))*(s + r*(Xc - Xk)) < 0
        """
        super(cramming, self).__init__()
        self.model = model
        self.input_dim = self.model.layer_1.weight.data.shape[1]
        self.hidden_dim = self.model.layer_1.weight.data.shape[0]
        self.train_loader = train_loader
        self.X_train = X_train
        self.y_train = y_train

        self.lr_goal = lr_goal
        self.s = s
        self.ks = torch.where
    

    def cram(self):
        for i, k in enumerate(self.ks):
            print(f"cramming sample {k}th |{i/len(self.ks)*100:.2f}% total of {len(self.ks)}")
            r = self.cram_find_r(k)
            self.cram_add_node(r, k)
        torch.save(self.model, "acceptable/Cram.pth")

    def cram_find_r(self, k): 
        # L9, isolation R2: p39, carm: p.54, for multiple case: p.60?
        """
        k: k sample to cram (unaccepted sample with too large epsilon)
        ==========
        outputs
        r: vector that r*(Xc - Xk) != 0 and (s - r*(Xc - Xk))*(s + r*(Xc - Xk)) < 0
        """
        print("find r: r*(Xc - Xk) != 0 and (s - r*(Xc - Xk))*(s + r*(Xc - Xk)) < 0", end = "\r")

        Xc_Xk = []

        k = 1
        X_no_k = torch.cat([self.X_train[:k], self.X_train[k+1:]], dim = 0)
        if torch.any(torch.all(X_no_k == self.X_train[k], dim=1)):
            print("If X_train[k] in X_train: yes. check again")
        
        n = 0
        while True:
            n+=1
            print(f"try vector {n}", end="\r")

            r = torch.rand(self.input_dim)            
            dots = ((X_no_k - self.X_train[k]) @ r.T) 
            print(max((self.s + dots) * (self.s - dots)))
            
            if (torch.sum(dots == 0) == 0) and (max((self.s + dots) * (self.s - dots)) < 0):
                return r
            
    def cram_add_node(self, r, k):
        """
        k: k sample to cram (unaccepted sample with too large epsilon)
        r: vector that r*(Xc - Xk) != 0 and (s - r*(Xc - Xk))*(s + r*(Xc - Xk)) < 0
        """
        new_model = TwoLayerNet(self.input_dim, self.hidden_dim+3, 1)

        param = self.model.state_dict()
        for name in param:
            if name == 'layer_1.weight':
                # First node
                new_w = torch.cat([param[name], r.reshape(1, -1)], dim = 0)
                # Second node
                new_w = torch.cat([new_w, r.reshape(1, -1)], dim = 0)
                # Third node
                new_w = torch.cat([new_w, r.reshape(1, -1)], dim = 0)

                new_model.layer_1.weight.data = new_w
                    
            if name == 'layer_1.bias':
                # First node
                node_add = self.s - torch.dot(r, self.X_train[k])
                new_b = torch.cat([param[name], node_add.reshape(1)], dim = 0)
                # Second node
                node_add = (-1) * torch.dot(r, self.X_train[k])
                new_b = torch.cat([new_b, node_add.reshape(1)], dim = 0)
                # Third node
                node_add = (-1)*self.s - torch.dot(r, self.X_train[k])
                new_b = torch.cat([new_b, node_add.reshape(1)], dim = 0)

                new_model.layer_1.bias.data = new_b

            if name == 'layer_out.weight':
                """
                # the base of Xk = (yk - prediction yk)/s
                aik = nn.ReLU()(model.layer_1.weight.data @ X_train[k].T)
                out_weight = model.layer_out.weight.data.reshape(-1)
                out_bias = model.layer_out.bias.data.reshape(-1)
                base = (y_train[k] - out_bias - torch.dot(out_weight, aik))/s
                """
                base = (self.y_train[k] - self.model(self.X_train[k]))/self.s
                # First node
                new_w_o = torch.cat([param[name], base.reshape(1, 1)], dim = 1)
                # Second node
                new_w_o = torch.cat([new_w_o, ((-2)*base).reshape(1, 1)], dim = 1)
                # Third node
                new_w_o = torch.cat([new_w_o, base.reshape(1, 1)], dim = 1)

                new_model.layer_out.weight.data = new_w_o

        self.model = new_model
        self.hidden_dim +=3