"""
Utilizing the default parameters implemented in PyTorch.

    Args:
        data [Tensor]: input images with a size of [Batch size, Channel, Height, Width]
        seed [int]: seed for randomization; default: "6759" 

    Returns:
        aug_data [Tensor]: output images with a size of [4*Batch size, Channel, Height, Width], where the first [Batch size, Channel, Height, Width] is the original data

"""

import torch
import torchvision
import torchvision.transforms as T

def Aug(data,labels):
       
    # To allow the method to perform properly, need to inverse the normalization performed first.
    inv_normalize = T.Normalize(
        mean=[-0.49139968/0.24703223, -0.48215841/0.24348513, -0.44653091/0.26158784],
        std=[1/0.24703223, 1/0.24348513, 1/0.26158784])
    
    #torch.manual_seed(seed)
    inv_norm_data = inv_normalize(data)*255
    aug_data = inv_norm_data
    RandAug = T.RandAugment()
    
    for i in range(1):        
            aug_data = torch.cat((aug_data,RandAug(((inv_norm_data).to(torch.uint8))) ))
    
    # Reapplying the normalization
    normalize = T.Normalize([0.49139968, 0.48215841, 0.44653091], [0.24703223, 0.24348513, 0.26158784])
    
    aug_data = normalize(aug_data)
    
    aug_labels = torch.cat((labels,labels))
    
    return aug_data,aug_labels