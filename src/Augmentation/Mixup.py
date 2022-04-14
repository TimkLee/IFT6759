"""
Select a random image and overlay the two images. A strength parameter is used to adjust the pixel values of the two images (x_new = S*x_i + (1-S)*x_j). The final label is adjusted based on the patch size.
The strength parameter is selected in the range of [0, 1].

    Args:
    data [Tensor]: input images with a size of [Batch size, Channel, Height, Width]
    labels [Tensor]: input labels with a size of [Batch size, ]
    seed [int]: seed for randomization; default: "6759"

    Returns:
    aug_data [Tensor]: output images with a size of [2*Batch size, Channel, Height, Width]
    aug_labels [Tensor]: onehot encoded output labels with a size of [2*Batch size,]

"""
import torch
import numpy as np

def Aug(data,labels):
    #torch.manual_seed(seed)
    alpha = 0.4

    #sample alpha from beta distribution
    get_alpha = np.random.beta(alpha, alpha, labels.size(0))
    get_alpha = np.concatenate([get_alpha[:,None], 1-get_alpha[:,None]], 1).max(1)
    get_alpha = data.new(get_alpha)
    get_alpha = get_alpha.unsqueeze(1)

    #shuffle batch to get the second image to augment
    shuffle = torch.randperm(data.size(0)).to(data.device)
    x1, y1 = data[shuffle], labels[shuffle]
    
    aug_data = (data * get_alpha.view(get_alpha.size(0),1,1,1) + x1 * (1-get_alpha).view(get_alpha.size(0),1,1,1))
    aug_labels = labels * get_alpha + y1 * (1-get_alpha)

    aug_data = torch.cat((data, aug_data))
    aug_labels = torch.cat((labels, aug_labels))
    
    return aug_data , aug_labels
