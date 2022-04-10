"""
Operation is performed using the `gaussian_blur` function in PyTorch, the `kernel_size` is selected in the range of [1,15].

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
    aug_data = data
    
    for i in range(1):        
            aug_data = torch.cat((aug_data,F.gaussian_blur(data,kernel_size=[2*int(torch.randint(0, 7,(1,)))+1,2*int(torch.randint(0, 7,(1,)))+1])))

    aug_labels = torch.cat((labels,labels))
    
    return aug_data,aug_labels