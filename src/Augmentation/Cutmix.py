"""
Apply a single rectangular mask to the image. The portion of the mask that lies outside the image is ignored. Fill the masked area with pixels located at the same area from another randomly selected image. The final label is adjusted based on the patch size.
Randomly select a pixel within the image, this pixel represents the center of the mask. 
Select a value in the range of [1, 8], this value times 2 represents the width of the mask.
Select a value in the range of [1, 8], this value times 2 represents the height of the mask.


    Args:


    Returns:

"""

def Aug():
    
    return print("Augmented")