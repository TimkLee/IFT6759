"""
Apply horizontal flip, a vertical flip, or both at the same time with equal probability.

    Args:
        data [Tensor]: input images with a size of [Batch size, Channel, Height, Width]
        seed [int]: seed for randomization; default: "6759" 

    Returns:
        aug_data [Tensor]: output images with a size of [4*Batch size, Channel, Height, Width], where the first [Batch size, Channel, Height, Width] is the original data

"""

import torch
import torchvision
import torchvision.transforms.functional as F

def Aug(data,seed = 6759):
    
    torch.manual_seed(seed)
    prob = torch.rand(3)
    aug_data = data
    
    for i in range(len(prob)):
        
        if prob[i]<0.3333: 
            aug_data = torch.cat((aug_data,F.vflip(data)))
            
        elif prob[i]>0.6666:
            aug_data = torch.cat((aug_data,F.hflip(data)))
        
        else:
            temp = F.vflip(data)
            aug_data = torch.cat((aug_data,F.hflip(temp)))
    
    return aug_data