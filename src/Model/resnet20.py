from torchvision.models import resnet18
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class ModelClass(nn.Module):


    def __init__(self, num_classes=10,optimizer = "adam", lr=0.003, weight_decay = 0.01, momentum = 0.9,criterion = "CrossEntropyLoss"):
        """
        Reference : https://pytorch-tutorial.readthedocs.io/en/latest/tutorial/chapter03_intermediate/3_2_2_cnn_resnet_cifar10/
        """
        super(ModelClass,self).__init__()
               
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(ResidualBlock, out_channels=16, blocks = 3)
        self.layer2 = self.make_layer(ResidualBlock, out_channels=32, blocks = 3, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, out_channels=64, blocks = 3, stride=2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64, num_classes)        
        
        if optimizer == "adam":
            self.optimizer = optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer == "sgd":
            self.optimizer = optim.SGD(self.parameters(), lr=lr,momentum = momentum, weight_decay=weight_decay)
        else:
            print("Optimizer not recognized")

        if criterion == "CrossEntropyLoss":
            self.criterion = nn.CrossEntropyLoss()    
        else:
            print("Loss criterion not recognized")
            
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        input : x of shape (batch size, channel , height, width)
        output : y_pred, output after forward pass into resnet18 block. With shape 
        """

        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
  

    
    #def train_sup_up(self, epoch, dataloader, optimizer, criterion):
    def train_sup_up(self, data, target):
        """
        TRAIN LOOP FOR SUPERVISED/UNSUPERVISED LEARNING
        Train the model with the given criterion, optimizer, dataloader for given epochs. Results propogated to WANDB for visualization
        """
        #TODO
        
        self.train()
        self.optimizer.zero_grad()
        y_pred = self.forward(data)

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
        self.eval()
        with torch.no_grad():

            y_pred = self.forward(data)

            loss = self.criterion(y_pred,target)
            
            acc = (torch.argmax(y_pred, dim=1) == torch.argmax(target, dim=1)).float().sum()/target.shape[0] 
        
        return acc, loss



    def update_lr(self, lr):    
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr




def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
