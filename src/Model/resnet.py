from torchvision.models import resnet18
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class ModelClass(nn.Module):

    def __init__(self,num_classes=10,optimizer = "adam", lr=0.003, weight_decay = 0.01, criterion = "CrossEntropyLoss"):
        """
        reuse of torchvision module : https://pytorch.org/vision/stable/generated/torchvision.models.resnet18.html
        """
        super(ModelClass,self).__init__()
        self.resnet18 = resnet18(progress=False,num_classes=num_classes)
        
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.resnet18.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            print("Optimizer not recognized")

        if criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()    
        else:
            print("Loss criterion not recognized")
            
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input : x of shape (batch size, channel , height, width)
        output : y_pred, output after forward pass into resnet18 block. With shape 
        """
        #with torch.no_grad():
        self.resnet18.eval()
        with torch.no_grad():
            # y_pred = F.softmax(self.resnet18(data))
            # y_pred = self.resnet18(data)
            y_pred = torch.sigmoid(self.resnet18(data))
            
        return y_pred

    
    #def train_sup_up(self, epoch, dataloader, optimizer, criterion):
    def train_sup_up(self, data, target):
        """
        TRAIN LOOP FOR SUPERVISED/UNSUPERVISED LEARNING
        Train the model with the given criterion, optimizer, dataloader for given epochs. Results propogated to WANDB for visualization
        """
        #TODO
        
        self.resnet18.train()
        self.optimizer.zero_grad()
        # y_pred = self.resnet18(data)
        y_pred = torch.sigmoid(self.resnet18(data))
        
        loss = self.criterion(y_pred,target)
        loss.backward()
        self.optimizer.step()
        
        acc = (torch.argmax(y_pred, dim=1) == torch.argmax(target, dim=1)).float().sum()/target.shape[0]      

        return acc, loss

    
    def train_shot(self, epoch, dataloader, optimizer, criterion):
        """
        TRAIN LOOP FOR FEWSHOT/ZEROSHOT LEARNING
        Train the model with the given criterion, optimizer, dataloader for given epochs. Results propogated to WANDB for visualization
        """
        #TODO
        pass


    #@torch.no_grad()
    def evaluation(self, data, target):
        """
        Evaluate the model for various evaluation metrics. Results propogated to WANDB for visualization 
        """
        #TODO
        self.resnet18.eval()
        with torch.no_grad():
            # y_pred = self.resnet18(data)
            y_pred = torch.sigmoid(self.resnet18(data))
            
            loss = self.criterion(y_pred,target)
            
            acc = (torch.argmax(y_pred, dim=1) == torch.argmax(target, dim=1)).float().sum()/target.shape[0] 
        
        return acc, loss


