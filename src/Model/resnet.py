from torchvision.models import resnet18
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelClass(nn.Module):

    def __init__(self,num_classes=10):
      """
      reuse of torchvision module : https://pytorch.org/vision/stable/generated/torchvision.models.resnet18.html
      """
      super(ModelClass,self).__init__()
      self.resnet18 = resnet18(progress=False,num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
      """
      input : x of shape (batch size, channel , height, width)
      output : y_pred, output after forward pass into resnet18 block. With shape 
      """
      y_pred = self.resnet18(x)
      return y_pred

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


