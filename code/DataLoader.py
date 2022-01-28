import os
import pickle
import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(data_dir):
    """Load the CIFAR-10 dataset.

    Args:
        data_dir: A string. The directory where data batches
            are stored.

    Returns:
        x_train: An numpy array of shape [50000, 3072].
            (dtype=np.float32)
        y_train: An numpy array of shape [50000,].
            (dtype=np.int32)
        x_test: An numpy array of shape [10000, 3072].
            (dtype=np.float32)
        y_test: An numpy array of shape [10000,].
            (dtype=np.int32)
    """

    ### YOUR CODE HERE
    train_files = [
    'data_batch_1',
    'data_batch_2',
    'data_batch_3',
    'data_batch_4',
    'data_batch_5',
    ]
    test_files = ['test_batch']

    x_train = np.empty((0,3072), int)
    y_train = np.empty((0))
    for i in train_files:
        with open(os.path.join(data_dir, i), 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3072)
            Y = np.array(Y)
        x_train = np.append(x_train, X, axis=0)
        y_train = np.append(y_train, Y, axis=0)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    
    x_test = np.empty((0,3072), int)
    y_test = np.empty((0))
    for i in test_files:
        with open(os.path.join(data_dir, i), 'rb') as f:
            datadict = pickle.load(f, encoding='latin1')
            X = datadict['data']
            Y = datadict['labels']
            X = X.reshape(10000,3072)
            Y = np.array(Y)
        x_test = np.append(x_test, X, axis=0)
        y_test = np.append(y_test, Y, axis=0)

    x_test = np.array(x_test)
    y_test = np.array(y_test)
    ### END CODE HERE

    return x_train, y_train, x_test, y_test


def load_testing_images(data_dir):
    """Load the images in private testing dataset.

    Args:
        data_dir: A string. The directory where the testing images
        are stored.

    Returns:
        x_test: An numpy array of shape [N, 32, 32, 3].
            (dtype=np.float32)
    """

    ### YOUR CODE HERE
    x_test = np.load(data_dir)
    print("Private Test Data Shape")
    print(x_test.shape)
    print()
    ### END CODE HERE

    return x_test


def train_valid_split(x_train, y_train, train_ratio=0.8):
    """Split the original training data into a new training dataset
    and a validation dataset.

    Args:
        x_train: An array of shape [50000, 3072].
        y_train: An array of shape [50000,].
        train_ratio: A float number between 0 and 1.

    Returns:
        x_train_new: An array of shape [split_index, 3072].
        y_train_new: An array of shape [split_index,].
        x_valid: An array of shape [50000-split_index, 3072].
        y_valid: An array of shape [50000-split_index,].
    """
    
    ### YOUR CODE HERE
    split_index = int(x_train.shape[0] * train_ratio)
    x_train_new = x_train[:split_index]
    y_train_new = y_train[:split_index]
    x_valid = x_train[split_index:]
    y_valid = y_train[split_index:]
    ### END CODE HERE

    return x_train_new, y_train_new, x_valid, y_valid

