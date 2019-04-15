import h5py
import numpy as np

class Dataset(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.store = h5py.File(dataset_path)
        self.dataset_names = list(self.store.keys())
        self.datasets = [self.store[dataset_name] for dataset_name in self.dataset_names]
        self.dataset_shapes = [dataset.shape for dataset in self.datasets]

    def random_iterator(self, batch_size, sequence_length, n_batches=None, rng=None):
        if rng is None:
            rng = np.random.RandomState()

        # Determine how many batches there are per sequence per dataset
        batch_indices = []
        for i, shape in enumerate(self.dataset_shapes):
            n, l, c = shape  # Number of sequences, length of sequences and number of channels
            for j in range(n):
                for k in range(l//sequence_length):
                    batch_indices.append((i, j, k))
        rng.shuffle(batch_indices)
        if n_batches is None:
            n_batches = len(batch_indices) // batch_size

        datasets = self.datasets
        for batch_i in range(n_batches):
            start = batch_i*batch_size
            end = start + batch_size
            indices = batch_indices[start: end]
            data = np.array([datasets[i][j, k*sequence_length: (k+1)*sequence_length] for i,j,k in indices])
            yield data





