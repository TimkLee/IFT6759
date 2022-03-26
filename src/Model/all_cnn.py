import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelClass(nn.Module):
    """
    All CNN model C 
    ref : https://arxiv.org/pdf/1412.6806.pdf check table 2 (we replace max pooling from table 1 with conv layers with stride=2 to get AllCNN correctly)
    """
    def __init__(self, num_classes=10):
        super(ModelClass, self).__init__()
        
        self.num_classes = num_classes
        
        #Convolutional Layers:
        #Block1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(3, 3), padding='same')
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), padding='same')
        #BLock2
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=(3, 3), padding='same')
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding='same')
        #Block3
        self.conv5 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), padding='same')
        #Block4
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(1, 1), padding='valid')
        #Block5
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=self.num_classes, kernel_size=(1, 1), padding='valid')

        #MaxPool Layers
        #Block1
        self.max1 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        #Block2
        self.max2 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

        #GlobalAveragePooling
        #Useful Reference: https://discuss.pytorch.org/t/global-average-pooling-in-pytorch/6721/12
        self.gap = nn.AvgPool2d(kernel_size=(6, 6))

    def forward(self, x):
        """
        input : x of shape (batch size, channel , height, width)
        output after forward of dimension (batch_size , num_class)
        """
        #Block1
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max1(x)
        #Block2
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.max2(x)
        #Block3
        x = F.relu(self.conv5(x))
        #Block4
        x = F.relu(self.conv6(x))
        #Block5
        x = F.relu(self.conv7(x))
        #GlobalAveragePool
        x = self.gap(x)
        x = x.squeeze()
        return x

    def train_sup_up(self, epoch, dataloader, optimizer, criterion):
        """
        TRAIN LOOP FOR SUPERVISED/UNSUPERVISED LEARNING
        Train the model with the given criterion, optimizer, dataloader for given epochs. Results propogated to WANDB for visualization
        """
        #TODO
        pass

    def train_shot(self, epoch, dataloader, optimizer, criterion):
        """
        TRAIN LOOP FOR FEWSHOT/ZEROSHOT LEARNING
        Train the model with the given criterion, optimizer, dataloader for given epochs. Results propogated to WANDB for visualization
        """
        #TODO
        pass


    @torch.no_grad()
    def evaluation():
        """
        Evaluate the model for various evaluation metrics. Results propogated to WANDB for visualization 
        """
        #TODO
        pass