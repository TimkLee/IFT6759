"""
Apply a single rectangular mask to the image. The portion of the mask that lies outside the image is ignored.
Randomly select a pixel within the image, this pixel represents the center of the mask. 
Select a value in the range of [1, 8], this value times 2 represents the width of the mask.
Select a value in the range of [1, 8], this value times 2 represents the height of the mask.


    Args:


    Returns:

"""

import torch
import numpy as np


def Aug(data, labels):
    # torch.manual_seed(seed)

    # setting seed here would cutout the same portion of the image everytime. which is not the intended behaviour
    # TODO: reproducibility vs algo tradeoff
    #np.random.seed(seed)

    h = data.shape[2]
    w = data.shape[3]
    
    # Using default settings in official implementation: 
    n_holes = 1
    length = 16

    aug_data = data
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for i in range(1):
        temp = data
        for ind, img in enumerate(data):
            mask = torch.ones(h,w, dtype = torch.float32)
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

            mask = mask.expand_as(img)
            mask = mask.to(device)
            temp[ind] = img * mask
        aug_data = torch.cat((aug_data,temp))
    
    aug_labels = torch.cat((labels,labels))
    
    return (aug_data, aug_labels)
