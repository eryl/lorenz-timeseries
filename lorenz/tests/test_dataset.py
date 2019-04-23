import unittest
import os

import matplotlib.pyplot as plt
import numpy as np

from lorenz.lorenz import make_dataset, plot_3d
import lorenz.dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], 'data', 'lorenz.h5')

    def test_random_iterator_1d(self):
        rng = np.random.RandomState(1729)
        dataset = lorenz.dataset.Dataset(self.path, view='1d')
        for b in dataset.random_iterator(4, 100):
            plt.plot(np.linspace(0,1,b.shape[1]), b[:,:,0].T)
            plt.show()
            #plot_3d(b)

    def test_random_iterator_3d(self):
        rng = np.random.RandomState(1729)
        dataset = lorenz.dataset.Dataset(self.path, view='3d')
        for b in dataset.random_iterator(4, 100):
            plot_3d(b)
