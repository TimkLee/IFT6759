""" 
Download the Cifar10 data and create the train, validation, test split based on the task of interest.

    Args:
        task [str]: learning task of interest; default: "super" 
        batch_size [int]: size of the data batch; default: "256" 
        seed [int]: seed for randomization; default: "6759" 

    Returns:
        labelledloader [torch.utils.data.DataLoader]: data loader for the training data with label
        unlabelledloader [torch.utils.data.DataLoader]: data loader for the training data without label, for supervised learning, this is equal to `labelledloader`
        validloader [torch.utils.data.DataLoader]: data loader for the validation data
        testloader [torch.utils.data.DataLoader]: data loader for the test data    
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import logging

logging.basicConfig(level=logging.INFO)

def Data_Load(task = "super",batch_size = 256, seed = 6759):
    
    # Transform images from PCI to tensors and normalize the data using Cifar10's mean and standard deviation
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])])    
    
    # Download the data
    traindata = CIFAR10(root='./Data/Cifar10_Data', train=True, download=True, transform=transform)
    testdata = CIFAR10(root='./Data/Cifar10_Data', train=False, download=True, transform=transform)
    logging.info("Data Extracted")
    
    # Note:
    # classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}
    
    # Save 10% of training data as validation data
    train_set, valid_set = random_split(traindata, [45000, 5000], generator=torch.Generator().manual_seed(seed))
    
    validloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=2)   
    testloader = DataLoader(testdata, batch_size=batch_size, shuffle=False, num_workers=2)
    
    if (task == "super"):
        # Supervised learning task
        logging.info("Data split for supervised learning task")

        labelledloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)   
        unlabelledloader = labelledloader
        
    elif (task == "semi"):
        # Semi-supervised learning task
        logging.info("Data split for semi-supervised learning task")
        # Use 20% of the data as "labelled" data
        label_set, unlabel_set = random_split(traindata, [9000, 36000], generator=torch.Generator().manual_seed(seed))
        labelledloader = DataLoader(label_set, batch_size=batch_size, shuffle=True, num_workers=2)  
        unlabelledloader = DataLoader(unlabel_set, batch_size=batch_size, shuffle=True, num_workers=2)  

    elif (task == "shot"):
        # Few-shot learning task
        logging.info("Data split for few-shot learning task")
        
        # 10-way 5-shots, search for the class and extract the data for the "labelled" and "unlabelled" dataset
        label_ind_list = torch.zeros(0)
        for i in range(10):
            target_ind = ((torch.Tensor(train_set.dataset.targets) == i).nonzero(as_tuple=False))
            label_ind_list = torch.cat((label_ind_list,target_ind[0:5]),0)        
        
        label_ind_list = label_ind_list.to(dtype=torch.long)
        label_set = torch.utils.data.Subset(train_set.dataset, label_ind_list)
        labelledloader = DataLoader(label_set, batch_size=256, shuffle=False, num_workers=2)
        
        templist = torch.arange(45000)
        mask = torch.ones(45000)
        mask[label_ind_list] = False
        unlabel_ind_list = templist[mask.to(torch.bool)]
        unlabel_set = torch.utils.data.Subset(train_set.dataset, unlabel_ind_list)
        unlabelledloader = DataLoader(unlabel_set, batch_size=256, shuffle=False, num_workers=2)
        
        
    else:
        logging.warning("Error: Task not recognized")
        sys.exit(1)
    
    
    logging.info("Data Loaded")
    
    return labelledloader, unlabelledloader, validloader, testloader


       
