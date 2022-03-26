"""
Select a random image and overlay the two images. A strength parameter is used to adjust the pixel values of the two images (x_new = S*x_i + (1-S)*x_j). The final label is adjusted based on the patch size.
The strength parameter is selected in the range of [0, 1].

    Args:


    Returns:

"""

def Aug(data,labels, seed = 6759):
    torch.manual_seed(seed)
    alpha = 0.4

    #sample alpha from beta distribution
    get_alpha = np.random.beta(alpha, alpha, labels.size(0))
    get_alpha = np.concatenate([get_alpha[:,None], 1-get_alpha[:,None]], 1).max(1)
    get_alpha = data.new(get_alpha)

    #shuffle batch to get the second image to augment
    shuffle = torch.randperm(data.size(0)).to(data.device)
    x1, y1 = data[shuffle], labels[shuffle]

    aug_data = (last_input * get_alpha.view(get_alpha.size(0),1,1,1) + x1 * (1-get_alpha).view(get_alpha.size(0),1,1,1))
    aug_labels = labels * get_alpha + y1 * (1-get_alpha)
    
    return (aug_data , aug_labels)
