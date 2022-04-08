"""
Apply a single rectangular mask to the image. The portion of the mask that lies outside the image is ignored. Fill the masked area with pixels located at the same area from another randomly selected image. The final label is adjusted based on the patch size.
Randomly select a pixel within the image, this pixel represents the center of the mask. 
Select a value in the range of [1, 8], this value times 2 represents the width of the mask.
Select a value in the range of [1, 8], this value times 2 represents the height of the mask.


    Args:
    data [Tensor]: input images with a size of [Batch size, Channel, Height, Width]
    labels [Tensor]: input labels with a size of [Batch size, ]
    seed [int]: seed for randomization; default: "6759"

    Returns:
    aug_data [Tensor]: output images with a size of [2*Batch size, Channel, Height, Width]
    aug_labels [Tensor]: onehot encoded output labels with a size of [2*Batch size, num_class,]

"""
import torch
import numpy as np

def onehot(size, target):
    vec = torch.zeros(size, dtype=torch.float32)
    vec[target] = 1.
    return vec

def rand_bbox(size, lam):
    if len(size) == 4:
        W = size[2]
        H = size[3]
    elif len(size) == 3:
        W = size[1]
        H = size[2]
    else:
        raise Exception

    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def Aug(data, labels, seed = 6759):
    torch.random.manual_seed(seed)
    np.random.seed(seed)

    beta=1.0
    prob=1.0
    num_class = 10

    aug_data = data
    aug_labels = labels

    for i, img in enumerate(data):
            lb = labels[i]
            lb_onehot = onehot(num_class, lb)

            r = np.random.rand(1)
            if beta <= 0 or r > prob:
                continue

            # generate mixed sample
            lam = np.random.beta(beta, beta)

            shuffle = torch.randperm(data.size(0)).to(data.device)
            img2, lb2 = data[shuffle], labels[shuffle]
            lb2_onehot = onehot(num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

            aug_data[i] = img
            aug_labels[i] = lb_onehot
    
    aug_data = torch.cat((data, aug_data))
    aug_labels = torch.cat((labels, aug_labels))

    return (aug_data, aug_labels)
