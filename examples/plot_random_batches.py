import os.path
import numpy as np
import matplotlib.pyplot as plt
import lorenz.dataset

from lorenz.lorenz import plot_3d

batch_size = 4
sequence_length = 100
n_batches = 3
path = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'lorenz_train.h5')
rng = np.random.RandomState(1729)

## We create two Datasets which wraps the same HDF5, one uses a 1D PCA projection, the other the original dataset
one_d_pca_dataset = lorenz.dataset.Dataset(path, view='pca', view_dims=0)
one_d_pca_dataset_noisy = lorenz.dataset.Dataset(path, view='pca', view_dims=0, noise=0.1)
full_original = lorenz.dataset.Dataset(path, view='original', view_dims=(0,1,2))
full_noisy_original = lorenz.dataset.Dataset(path, view='original', view_dims=(0,1,2), noise=0.05)

## Plot 3D trajectories
c = rng.randint(2**32-1)
rng_1 = np.random.RandomState(c)
rng_2 = np.random.RandomState(c)
full_noisy_iterator = full_noisy_original.random_iterator(batch_size, sequence_length, n_batches=n_batches, rng=rng_1)
full_iterator = full_original.random_iterator(batch_size, sequence_length, n_batches=n_batches, rng=rng_2)

for b1, b2 in zip(full_iterator, full_noisy_iterator):
    plot_3d(np.concatenate([b1, b2], axis=0))
    plt.show()

## Plot the one dimensional PCA projection of the dataset
c = rng.randint(2**32-1)
rng_1 = np.random.RandomState(c)
rng_2 = np.random.RandomState(c)
oned_noisy_iterator = one_d_pca_dataset_noisy.random_iterator(batch_size, sequence_length, n_batches=n_batches, rng=rng_1)
oned_iterator = one_d_pca_dataset.random_iterator(batch_size, sequence_length, n_batches=n_batches, rng=rng_2)

for b_noisy, b_clean in zip(oned_noisy_iterator, oned_iterator):
    plt.plot(np.linspace(0, 1, b_clean.shape[1]), b_clean.T)
    plt.plot(np.linspace(0, 1, b_noisy.shape[1]), b_noisy.T)
    plt.show()