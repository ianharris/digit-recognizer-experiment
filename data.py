import random
import pandas as pd
import numpy as np

"""
A class to provide batches of data to the training algorithm
"""
class Data():
    def __init__(self, uri):

        # initialise and index
        self.indices = []

        # read in csv
        # train = pd.read_csv('datasets/train.csv')
        # test = pd.read_csv('datasets/test.csv')
        data = pd.read_csv(uri)
        
        # create a labels and pixels dataframe
        if 'label' in data.columns:
            self.labels_set = True
            labels = data['label']
            pixels = data.drop('label', axis=1)
        else:
            self.labels_set = False
            pixels = data

        # create the numpy arrays
        self.features = np.reshape(pixels.as_matrix(), (-1, 28, 28, 1))

        if self.labels_set:
            self.labels = pd.get_dummies(labels).as_matrix()

    # TODO could end up getting one batch that is smaller than the requested size when indices
    # array nears empty - perhaps implement a recursvie call to the function to fix this
    def get_batch(self, batch_size=50):

        # if the set of indices hasn't been initialised or has been used up create it and shuffle
        if len(self.indices) == 0:
            self.indices = list(range(0, np.shape(self.features)[0]))
            random.shuffle(self.indices)

        # get the batches to return
        fbatch = self.features[self.indices[:batch_size], :, :, :]
        if self.labels_set:
            lbatch = self.labels[self.indices[:batch_size], :]
        else:
            lbatch = None

        # update the index with the num read
        self.indices = self.indices[np.shape(fbatch)[0]:] 
        return (fbatch, lbatch)

