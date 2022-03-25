import os
import random

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F

class AllCNN(nn.Module):
    """
    All CNN model C 
    ref : https://arxiv.org/pdf/1412.6806.pdf check table 2 (we replace max pooling from table 1 with conv layers with stride=2 to get AllCNN correctly)
    """
    def __init__(self, num_classes=10):
        super(AllCNN, self).__init__()
        self.kernel = (3,3)
        self.stride = (2,2)
        
        self.block_a = nn.Sequential(
            nn.Conv2d(3,96,kernel_size = self.kernel, padding="same"),
            nn.ReLU(),
            nn.Conv2d(96,96,kernel_size =self.kernel, padding="same"),
            nn.ReLU()
        )
        #replace maxpool of regular CNN with simple convolutional layer
        self.replace_max_pool_a = nn.Sequential(
            nn.Conv2d(96,96,kernel_size =self.kernel,stride=self.stride),
            nn.ReLU()
        )
        self.block_b = nn.Sequential(
            nn.Conv2d(96,192,kernel_size = self.kernel, padding="same"),
            nn.ReLU(),
            nn.Conv2d(192,192,kernel_size = self.kernel, padding="same"),
            nn.ReLU()
        )
        self.replace_max_pool_b = nn.Sequential(
            nn.Conv2d(192,192,kernel_size = self.kernel,stride=self.stride),
            nn.ReLU()
        )
        # Block D is the last part of the table 1 in a whole block but without max pooling
        self.block_d = nn.Sequential(
            nn.Conv2d(192,192,kernel_size = self.kernel),
            nn.ReLU(),
            nn.Conv2d(192,192,kernel_size = (1,1)),
            nn.ReLU(),
            nn.Conv2d(192,10,kernel_size = (1,1)),
            nn.ReLU(),
            # global avg and softmax is done in the forward
        )
        

    def forward(self, x):
        """
        input : x of shape (batch size, channel , height, width)
        output after forward of dimension (batch_size , num_class)
        """
        #blocks from all cnn
        print("In shape : ", x.shape)
        x=self.block_a(x)
        x=self.replace_max_pool_a(x)
        x=self.block_b(x)
        x=self.replace_max_pool_b(x)
        #last general block 
        x = self.block_d(x)
        #global avg pooling
        x = torch.mean(x,[2,3]) #ref :https://discuss.pytorch.org/t/how-can-i-perform-global-average-pooling-before-the-last-fully-connected-layer/74352/2
        #output fctn
        x = F.softmax(x)
        print("Out shape : ", x.shape)
        return x