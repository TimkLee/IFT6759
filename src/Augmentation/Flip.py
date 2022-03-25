"""
Apply horizontal flip, a vertical flip, or both at the same time with equal probability.

    Args:


    Returns:

"""



def Aug():
    
    
    
    
    
    hflipper = T.RandomHorizontalFlip(p=0.5)
    transformed_imgs = [hflipper(orig_img) for _ in range(4)]
    plot(transformed_imgs)
    
    vflipper = T.RandomVerticalFlip(p=0.5)
    transformed_imgs = [vflipper(orig_img) for _ in range(4)]
    plot(transformed_imgs)
    
    
    return print("Augmented")