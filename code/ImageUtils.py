import numpy as np
import torch

"""This script implements the functions for data augmentation
and preprocessing.
"""

def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing.

    Args:
        record: An array of shape [3072,]. One row of the x_* matrix.
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3].
    """
    ### YOUR CODE HERE
    
    # Reshape from [depth * height * width] to [depth, height, width].
    record = record/255
    depth_major = record.reshape((3, 32, 32))

    # Convert from [depth, height, width] to [height, width, depth]
    image = np.transpose(depth_major, [1, 2, 0])
    image = preprocess_image(image, training)

    # Convert from [height, width, depth] to [depth, height, width]
    image = np.transpose(image, [2, 0, 1])
    ### END CODE HERE

    return image


def preprocess_image(image, training):
    """Preprocess a single image of shape [height, width, depth].

    Args:
        image: An array of shape [32, 32, 3].
        training: A boolean. Determine whether it is in training mode.

    Returns:
        image: An array of shape [32, 32, 3]. The processed image.
    """
    ### YOUR CODE HERE
    if training:
        # Resize the image to add four extra pixels on each side.
        im_pad = np.pad(image, ((4,4), (4,4), (0,0))) 
        x = np.random.randint(0,8)
        y = np.random.randint(0,8)
        cropped_image = im_pad[x:x+32, y:y+32, :]

        # Randomly flip the image horizontally.
        should_flip = np.random.randint(2)
        if should_flip == 1:
            cropped_image = np.fliplr(cropped_image)
            
        image = cropped_image
    
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]
    
#     means = [0.4914, 0.4822, 0.4465]
#     stds = [0.2023, 0.1994, 0.2010]
    
#     for i in range(3):
#         image[:, :, i] = (image[:, :, i] - means[i])/stds[i]

    channels = [r_channel, g_channel, b_channel]
    standardized = []
    for img in channels:
        mean = np.mean(img)
        std = np.std(img)
        norm_img = (img - mean) / std
        standardized.append(norm_img)

    normalized_image = np.stack(standardized, axis=-1)
    standardized_image = np.array(normalized_image)
        
    image = standardized_image
        
    ### END CODE HERE

    return image


# Other functions
### YOUR CODE HERE
def get_mean_and_std(x):
    '''Compute the mean and std value of dataset.'''
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    x = np.array([rec.reshape((3, 32, 32)) for rec in x ])
    print('==> Computing mean and std..')
    for i in range(3):
        mean[i] += x[:,i,:,:].mean()
        std[i] += x[:,i,:,:].std()
    mean.div_(255)
    std.div_(255)
    return mean, std
### END CODE HERE