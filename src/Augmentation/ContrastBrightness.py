"""
Apply contrast adjustment, brightness adjustment, or both at the same time with equal probability. 
Contrast adjustment is performed using the `adjust_contrast` function in PyTorch, the `contrast_factor` is selected to be in the range of [0.5, 1.5].
Brightness adjustment is performed using the `adjust_brightness` function in PyTorch, the `brightness_factor` is selected to be in the range of [0.5, 1.5].

    Args:
        data [Tensor]: input images with a size of [Batch size, Channel, Height, Width]
        seed [int]: seed for randomization; default: "6759" 

    Returns:
        aug_data [Tensor]: output images with a size of [4*Batch size, Channel, Height, Width], where the first [Batch size, Channel, Height, Width] is the original data

"""

import torch
import torchvision
import torchvision.transforms.functional as F

def Aug(data,labels):
    
    #torch.manual_seed(seed)
    prob = torch.rand(1)
    data = data/255
    aug_data = data
    
    for i in range(len(prob)):
        
        if prob[i]<0.3333: 
            aug_data = torch.cat((aug_data,F.adjust_brightness(data, brightness_factor=torch.FloatTensor(1,).uniform_(0.5, 1.5))))
            
        elif prob[i]>0.6666:
            aug_data = torch.cat((aug_data,F.adjust_contrast(data, contrast_factor=torch.FloatTensor(1,).uniform_(0.5, 1.5))))
        
        else:
            temp = F.adjust_brightness(data, brightness_factor=torch.FloatTensor(1,).uniform_(0.5, 1.5))
            aug_data = torch.cat((aug_data,F.adjust_contrast(temp, contrast_factor=torch.FloatTensor(1,).uniform_(0.5, 1.5))))    
    
    aug_data = aug_data*255
    aug_labels = torch.cat((labels,labels))
    
    return aug_data,aug_labels