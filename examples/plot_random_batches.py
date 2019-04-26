import os.path
import numpy as np
import matplotlib.pyplot as plt
import lorenz.dataset

from lorenz.lorenz import plot_3d

path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'lorenz_train.h5')
rng = np.random.RandomState(1729)
dataset = lorenz.dataset.Dataset(path, view='1d')

for b in dataset.random_iterator(4, 100):
    plt.plot(np.linspace(0, 1, b.shape[1]), b[:, :, 0].T)
    plt.show()
    # plot_3d(b)

for b in dataset.random_iterator(4, 100):
    plot_3d(b)
