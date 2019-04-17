import unittest
import os

import numpy as np

from vindel.lorenz import make_dataset, plot_3d
import vindel.dataset

class TestDataset(unittest.TestCase):
    def setUp(self):
        self.path = os.path.join(os.path.split(os.path.split(os.path.dirname(__file__))[0])[0], 'data', 'lorenz.h5')


    def test_random_iterator(self):
        rng = np.random.RandomState(1729)
        dataset = vindel.dataset.Dataset(self.path)
        for b in dataset.random_iterator(4, 100):
            plot_3d(b)