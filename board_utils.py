'''
Util file for board generation

Key function is generate_[dim]_data() which does what it say on the tin.

An addiotional build_[dim]_dataset() is provided for convenience
'''

import numpy as np
from collections import namedtuple


class DataSet(object):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._num_samples = x.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def x(self):
        return self._x

    @property
    def y(self):
        return self._y

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size):
        """Return the next `batch_size` samples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_samples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_samples)
            np.random.shuffle(perm)
            self._x = self._x[perm]
            self._y = self._y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_samples
        end = self._index_in_epoch
        return self._x[start:end], self._y[start:end]


def build_1d_datasets(
        data=None,
        labels=None,
        train_split=0.8,  # fraction of full dataset for training
        valid_split=0.5,  # fraction of test set for validation
        **kwargs):

    # Create dataset named tuple
    Datasets = namedtuple('Datasets', ['train', 'valid', 'test'])

    # Split dataset
    num_train = int(len(data) * train_split)
    num_valid = int((len(data) - num_train) * valid_split)
    
    Datasets.train = DataSet(data[:num_train], labels[:num_train])
    Datasets.valid = DataSet(data[:num_valid], labels[:num_valid])
    Datasets.test = DataSet(data[:num_valid], labels[:num_valid])

    return Datasets


def generate_1d_data(
        n_length=8,  # length of the grid
        num_samples=100000,
        k_value=2,  # number of colours k=2: binary
        one_hot=False,
        verbose=False):
    '''build 1d board training dataset of num_samples with n length
    for machine learning.
    
    The boards are randomly distributed binary "stones" (1 or 0)
    
    inputs:
        n: length of the board (1d)
        num_samples: how many samples
        k_value: number of colors of the grid default k=2 is binary
        train_split: 
        valid_split: 
        verbose: show statistics
        
    returns:
        A named tuple 'Dataset' with train, valid and test datasets
            of (data, labels) length
    '''
    
    # Generate random num_samples with n length in shape [samples, grid length, input dims]
    # It generates the random distribution from 0 to k_value
    # data = np.random.randint(0, k_value, size=[num_samples, n_length, k_value - 1])
    data = np.random.randint(0, k_value, size=[num_samples, n_length])
    
    # Generate labels from data
    # The arbitrary problem for the machine is to find the
    # connection length from left to right.

    labels = (np.zeros([num_samples, n_length], dtype=int) if one_hot
        else np.zeros(num_samples, dtype=int))
    for i, board in enumerate(data):

        #         if np.sum(board, axis=0) == n:  # quickly get fully connected boards
        #             labels[i] = 1
        #         else:
        #             labels[i] = 0

        # Stepwise look for 1's per grid
        connection_length = 0
        for j, grid in enumerate(board):
            if grid == 1:
                connection_length += 1
            else:
                break  # stop looking to save some computation
        if one_hot:
            if connection_length:
                labels[i][connection_length - 1] = 1
        else:
            labels[i] = connection_length  # one-hot

    return data, labels
