import h5py
import numpy as np

class Dataset(object):
    def __init__(self, dataset_path, view='pca', view_dims=(0,), noise=0):
        """
        Create a new dataset handler for Lorenz data.
        :param dataset_path: The path to the HDF5 dataset.
        :param view: Which view to use, 'pca' or 'original'. 'pca' contains the PCA projected view on the data, where
                     the order of the basis vectors where ordered by the eigenvalues (so the first dimension correspond
                     to the leading eigenvalue.
        :param view_dims: The dimensions to view the data in, by default only the leading dimension is used, resulting
                          in a 1D principal view of the data. The tuple (0,1,2) would instead give the full data.
        :param noise: The amount of noise to add to the observation, this corresponds to the standard deviation of the
                      Gaussian noise added to each sample.
        """
        self.view = view
        self.view_dims = view_dims
        self.noise = noise
        self.dataset_path = dataset_path
        self.store = h5py.File(dataset_path)
        self.group_names = list(self.store.keys())
        self.datasets = []
        for group_name, group in sorted(self.store.items()):
            for dataset_group_name, dataset_group in sorted(group.items()):
                group_datasets = [dataset for name, dataset in dataset_group.items() if view in name]
                if not group_datasets:
                    print("No datasets matching the view {} in group {}/{}".format(view, group_name, dataset_group_name))
                self.datasets.extend(group_datasets)
        self.dataset_shapes = [dataset.shape for dataset in self.datasets]

    def get_n_dims(self):
        return len(self.view_dims)

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
        else:
            n_batches = min(len(batch_indices) // batch_size, n_batches)

        datasets = self.datasets
        for batch_i in range(n_batches):
            start = batch_i*batch_size
            end = start + batch_size
            indices = batch_indices[start: end]
            data = np.array([datasets[i][j, k*sequence_length: (k+1)*sequence_length][:,self.view_dims] for i,j,k in indices])
            if self.noise > 0:
                data += rng.normal(scale=self.noise, size=data.shape)
            yield data





