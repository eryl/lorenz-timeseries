import os
import functools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This needs to be imported for the projection='3d' to work in plot_3d
import numpy as np
from scipy import integrate
import h5py

# We really want all data to use the same principal components, otherwise it'll be even more impossible to
# differentiate between the different hyper parameters and starting conditions. This is where we calculate those
PRINCIPAL_COMPONENTS_PATH = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'standardize_statistics.npz')
def _calculate_canonical_statistics():
    if os.path.exists(PRINCIPAL_COMPONENTS_PATH):
        principal_components = np.load(PRINCIPAL_COMPONENTS_PATH)
        return principal_components
    print("Generating canonical summary statistics and principal components, this might take a couple of minutes")
    n = 1000
    t = 10
    t_skip = 1
    dt = 0.01
    rng_seed = 1729
    noisy_hp_trajectories = [y_t_s.reshape((-1, 3)) for y_t_s, _ in generate_lorenz_trajectories(n, t, dt, t_skip,
                                                                                noise_level=0.1, rng_seed=rng_seed)]
    constant_hp_trajectories = [y_t_s.reshape(-1, 3) for y_t_s, _ in generate_lorenz_trajectories(n, t, dt, t_skip,
                                                                                   noise_level=0, rng_seed=rng_seed)]
    data = np.concatenate(noisy_hp_trajectories + constant_hp_trajectories, axis=0)
    m = data.mean(axis=0, keepdims=True)
    centered_data = data - m
    std = centered_data.std(axis=0, keepdims=True)
    standardized_data = centered_data / std
    eig_val, eig_vec = np.linalg.eigh(np.cov(standardized_data, rowvar=False))
    eig_order = np.argsort(eig_val)[::-1]
    principal_components = eig_vec[:,eig_order]
    principal_values = eig_val[eig_order]

    os.makedirs(os.path.dirname(PRINCIPAL_COMPONENTS_PATH), exist_ok=True)
    np.savez(PRINCIPAL_COMPONENTS_PATH, pc=principal_components, pv=principal_values, mean=m, std=std)
    print("Canonical principal components and summary statistics saved to {}".format(PRINCIPAL_COMPONENTS_PATH))
    return dict(pc=principal_components, pv=principal_values, mean=m, std=std)


def lorenz(x_y_z, t0, s=10., r=28., b=2.667):
    x, y, z = x_y_z
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def make_dataset(output, *args, mode='r+', **kwargs):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    stats = _calculate_canonical_statistics()
    mean = stats['mean']
    std = stats['std']
    pc = stats['pc']
    # A store consists of a three-deep nesting. The first level are Lorenz hyper parameter groups. Each entry in
    # such a group share the same hyper parameters for the Lorenz system (the rho, s and beta parameters) so they
    # are sampled trajectories of the same system.
    # The second level group is the hyper parameters for the sampling procedure: The total time, the random seed,
    # the time resolution and the time skip.
    # The final level are pairs of datasets, these are the original dataset with a name '{shape}-original' and the
    # PCA projection with the name '{shape}-pca', where {shape} is the shape of the respective datasets.
    with h5py.File(output, mode=mode) as store:
        trajectories = generate_lorenz_trajectories(*args, **kwargs)
        for y_ts, settings in trajectories:
            group_name = 'beta:{beta},rho:{rho},s:{s}'.format(**settings)
            group = store.require_group(group_name)
            group.attrs['rho'] = settings['rho']
            group.attrs['beta'] = settings['beta']
            group.attrs['s'] = settings['s']
            settings_group_name = 't:{t},dt:{dt},t_skip:{t_skip},rng_seed:{rng_seed}'.format(**settings)
            dataset_group = group.require_group(settings_group_name)
            dataset_group.attrs['t'] = settings['t']
            dataset_group.attrs['dt'] = settings['dt']
            dataset_group.attrs['t_skip'] = settings['t_skip']
            dataset_group.attrs['rng_seed'] = settings['rng_seed']
            y_ts = (y_ts - mean)/std  # Standardize the data
            dataset_group.create_dataset('{}-original'.format(y_ts.shape), data=y_ts)
            y_ts_pca = np.dot(y_ts, pc)
            dataset_pca = dataset_group.create_dataset('{}-pca'.format(y_ts_pca.shape), data=y_ts_pca)
            dataset_pca.attrs['pc'] = pc


def split_dataset(dataset_path, splits, normalize=False, rng=None):
    """
    Splits a lorenz hdf5 store into the splits defined by splits.
    :param dataset_path: The store to split
    :param splits: A list of (name, ratio) tuples, where the name will be suffixed to the original store path and the
                   ratio is the fraction of samples from the original store which ends up in this part.
    :param normalize: If set to True, the ratios will be normalized so that they sum to 1, allowing ratios to be
                      specified in other proportions (e.g. [20, 80]).
    :param rng: A numpy random state to use for sampling the splits.
    :return: None. The splits will be written to new HDF5 files using the split names.
    """
    if rng is None:
        rng = np.random.RandomState()
    split_names, ratios = zip(*splits)
    ratios_sum = sum(ratios)
    if ratios_sum < 1.:
        print("Warning: The ratios sum to {} < 1. The remaining trajectories will be unused.")
        # If not all of the dataset is used (the ratios sum to a value less than 1, we add a dummy
        # ratio
        split_names.append(None)
        ratios.append(1. - ratios_sum)
        ratios_sum = 1.
    if ratios_sum > 1.:
        if normalize:
            ratios = [r / ratios_sum for r in ratios]
        else:
            raise ValueError("The ratios sum to a value greater than 1, use --normalize if this is intentional")

    with h5py.File(dataset_path) as original_store:
        # A store consists of a three-deep nesting. The first level are Lorenz hyper parameter groups. Each entry in
        # such a group share the same hyper parameters for the Lorenz system (the rho, s and beta parameters) so they
        # are sampled trajectories of the same system.
        # The second level group is the hyper parameters for the sampling procedure: The total time, the random seed,
        # the time resolution and the time skip.
        # The final level are pairs of datasets, these are the original dataset with a name '{shape}-original' and the
        # PCA projection with the name '{shape}-pca', where {shape} is the shape of the respective datasets.
        # The splits will sample evenly from all groups
        store_base_name, ext = os.path.splitext(dataset_path)
        split_stores = []
        for split_name in split_names:
            if split_name is None:
                continue
            path = store_base_name + '_' + split_name + ext
            split_stores.append(h5py.File(path, 'w'))

        for system_group_name, system_group in original_store.items():
            for store in split_stores:
                g = store.create_group(system_group_name)
                g.attrs.update(system_group.attrs)

            for ode_settings_name, ode_settings in system_group.items():
                ode_settings_path = system_group_name + '/' + ode_settings_name
                if len(ode_settings) != 2:
                    raise NotImplementedError("The splitting is not implemented for settting groups which have more than "
                                              "two datasets. Group in question: {}".format(ode_settings_path))
                for store in split_stores:
                    g = store.create_group(ode_settings_path)
                    g.attrs.update(ode_settings.attrs)

                original_dataset_name, original_dataset = None, None
                pca_dataset_name, pca_dataset = None, None
                for dataset_name, dataset in ode_settings.items():
                    if 'original' in dataset_name:
                        original_dataset_name, original_dataset = dataset_name, dataset
                    elif 'pca' in dataset_name:
                        pca_dataset_name, pca_dataset = dataset_name, dataset
                assert original_dataset is not None and pca_dataset is not None, "Could not find the datasets"
                n_trajectories = original_dataset.shape[0]
                if n_trajectories < len(ratios):
                    raise ValueError("Dataset {} doesn't have enough trajectories for split".format(system_group_name +
                                                                                                    '/' + ode_settings_name
                                                                                                    + '/' + original_dataset))
                split_ns = [int(np.floor(r*n_trajectories)) for r in ratios[:-1]]
                split_ns.append(n_trajectories - sum(split_ns))
                # Now redistribute if there are empty splits
                while True:
                    max_n_index = np.argmax(split_ns)
                    min_n_index = np.argmin(split_ns)
                    if split_ns[min_n_index] == 0:
                        split_ns[max_n_index] -= 1
                        split_ns[min_n_index] += 1
                    else:
                        break

                assert sum(split_ns) == original_dataset.shape[0]
                indices = np.arange(original_dataset.shape[0])
                rng.shuffle(indices)
                start = 0
                for n, split_store in zip(split_ns, split_stores):
                    end = start + n
                    split_indices = indices[start:end]
                    split_data_original = original_dataset[:][split_indices]
                    split_data_pca = pca_dataset[:][split_indices]
                    ds_original = split_store[ode_settings_path].create_dataset(original_dataset_name, data=split_data_original)
                    ds_pca = split_store[ode_settings_path].create_dataset(pca_dataset_name, data=split_data_pca)
                    ds_original.attrs.update(original_dataset.attrs)
                    ds_pca.attrs.update(pca_dataset.attrs)
                    start = end


        for store in split_stores:
            store.close()



def generate_lorenz_trajectories(n, t, dt,
                                 t_skip=0,
                                 lorenz_beta=8/3.,
                                 lorenz_s=10.,
                                 lorenz_rho=28.,
                                 noise_level=0.,
                                 n_per_perturbance=1,
                                 rng_seed=None):

    rng = np.random.RandomState(rng_seed)

    if noise_level > 0:
        n_perturbances = int(np.ceil(n / n_per_perturbance))
        for i in range(n_perturbances):
            noisy_lorenz_beta = lorenz_beta + noise_level * rng.randn() * (8 / 3)
            noisy_lorenz_s = lorenz_s + noise_level * rng.randn() * 10
            noisy_lorenz_rho = lorenz_rho + noise_level * rng.randn() * 28
            local_rng_seed = rng.randint(2 ** 32)
            y_t = generate_trajectories(n_per_perturbance,
                                        t,
                                        t_skip=t_skip,
                                        dt=dt,
                                        lorenz_beta=noisy_lorenz_beta,
                                        lorenz_s=noisy_lorenz_s,
                                        lorenz_rho=noisy_lorenz_rho,
                                        rng=np.random.RandomState(local_rng_seed))
            settings = dict(beta=noisy_lorenz_beta, rho=noisy_lorenz_rho, s=noisy_lorenz_s, dt=dt, t=t,
                            t_skip=t_skip, rng_seed=local_rng_seed)
            yield (y_t, settings)
    else:
        local_rng_seed = rng.randint(2 ** 32)
        y_t = generate_trajectories(n,
                                    t,
                                    t_skip=t_skip,
                                    dt=dt,
                                    lorenz_beta=lorenz_beta,
                                    lorenz_s=lorenz_s,
                                    lorenz_rho=lorenz_rho,
                                    rng=np.random.RandomState(local_rng_seed))
        settings = dict(beta=lorenz_beta, rho=lorenz_rho, s=lorenz_s, dt=dt, t=t, t_skip=t_skip,
                        rng_seed=local_rng_seed)
        yield (y_t, settings)


def generate_trajectories(n,
                          t_max,
                          t_skip=0,
                          dt=0.01,
                          initial_value_ranges=((-20., -30., 0.0), (20., 30., 50.)),
                          lorenz_beta=8/3.,
                          lorenz_s=10.,
                          lorenz_rho=28.,
                          rng=None):
    if rng is None:
        rng = np.random.RandomState()
    a = np.array(initial_value_ranges[0])[np.newaxis,:]  # Lower bound for the uniform box of initial values
    b = np.array(initial_value_ranges[1])[np.newaxis,:]  # Upper bound for the uniform box of initial values
    y0s = (b - a) * rng.random_sample(size=(n, 3)) + a

    dts = int(t_max / dt)
    t = np.linspace(0, t_max, dts)
    y_t = np.zeros((n, dts, 3))

    f = functools.partial(lorenz, r=lorenz_rho, s=lorenz_s, b=lorenz_beta)
    for i in range(n):
        y0 = y0s[i]
        y_t[i] = integrate.odeint(f, y0, t)

    return y_t[:, int(t_skip/dt):, :]


def step(y0, rho, s, beta, t_max, dt):
    y_t = solve(y0, rho, s, beta, t_max, dt)
    return y_t[-1]


def solve(y0, rho, s, beta, t_max, dt):
    dts = int(t_max / dt)
    t = np.linspace(0, t_max, dts)

    f = functools.partial(lorenz, r=rho, s=s, b=beta)
    y_t = integrate.odeint(f, y0, t)

    return y_t

def dim_reduce_trajectories(data, n_components=None):
    """
    Reduce the dimensionality of the trajectories data using PCA.
    :param data: A 3d array with shape (num_trajectories, n, p), where n is the number of samples in the trajectory
                 and p number of features
    :param n_components: The number of principal components to use,
    :return: A 3d array of shape (num_trajectories, n, n_components)
    """

    principal_components = np.load(PRINCIPAL_COMPONENTS_PATH)['pc']
    if n_components is not None:
        principal_components = principal_components[:,:n_components]
    else:
        n_components = 3  # This is only set so that the below code for visual inspection of the results works

    # num_trajectories, n, dim = data.shape
    # all_data = data.reshape((num_trajectories * n, dim))
    # m = np.mean(all_data, axis=0)
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')
    # np.random.seed(1)
    # n_samples = min(num_trajectories*n, 1000)
    # data_sample_indices = np.random.choice(all_data.shape[0], size=n_samples, replace=False)
    # ax.scatter(all_data[data_sample_indices,0], all_data[data_sample_indices,1], all_data[data_sample_indices,2], alpha=0.5)
    # pca_line = np.zeros((2, 3, n_components), dtype=principal_components.dtype)
    # pca_line[0] = m[:,np.newaxis] - principal_components*30
    # pca_line[1] = m[:,np.newaxis] + principal_components*30
    # for i in range(n_components):
    #     x = pca_line[:, 0, i]
    #     y = pca_line[:, 1, i]
    #     z = pca_line[:, 2, i]
    #     ax.plot(x, y, z)
    # plt.show()
    # return
    pca_projection = np.dot(data, principal_components)
    return pca_projection, principal_components


def main():
    dt = 0.01
    n = 100  # Numbers of trajectories
    t_max = 10
    t_skip = 1
    lorenz_beta = 8 / 3
    lorenz_s = 10
    lorenz_rho = 28
    noise_level = 0.1

    y_ts, settings = zip(*generate_lorenz_trajectories(n, t_max, dt, t_skip=t_skip, lorenz_beta=lorenz_beta, lorenz_s=lorenz_s,
                                       lorenz_rho=lorenz_rho, noise_level=noise_level, rng_seed=1729))
    y_t = np.concatenate(y_ts)
    plot_3d(y_t)


def plot_channels(y_t, plot_dims=(0,1,2), pca_projection=None):
    num_channels, num_samples, dims = y_t.shape
    t = np.linspace(0, 1, num_samples)
    fig, axes = plt.subplots(num_channels, 1, squeeze=True)
    for i, ax in enumerate(axes):
        if plot_dims:
            plot_data = y_t[i, :, plot_dims]
            if pca_projection is not None:
                trajectory_pca_projection = pca_projection[i, :, np.newaxis]
                plot_data = np.concatenate([plot_data, trajectory_pca_projection], axis=-1)
        elif not pca_projection is None:
            plot_data = pca_projection[i, :, np.newaxis]
        else:
            raise ValueError("No plot_dims given, and no pca_projection data available.")
        ax.plot(t, plot_data)
    plt.show()


def plot_3d(y_t, pca_projection=None, pc1=None):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for i in range(y_t.shape[0]):
        y_t_i = y_t[i]
        ax.plot(y_t_i[:, 0], y_t_i[:, 1], y_t_i[:, 2], lw=0.5)
        if pca_projection is not None and pca_projection is not None:
            m = np.mean(y_t.reshape((-1, y_t.shape[-1])), axis=0, keepdims=True)
            projection_points = pca_projection[i] * pc1.T
            ax.scatter(projection_points[:,0], projection_points[:,1], projection_points[:,2])
            pca_line = np.zeros((3, 3, 1), dtype=pc1.dtype)
            pca_line[0] = m.T - pc1
            pca_line[1] = m.T
            pca_line[2] = m.T + pc1
            x = pca_line[:, 0, i]
            y = pca_line[:, 1, i]
            z = pca_line[:, 2, i]
            ax.plot(x, y, z, '.-')

    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")


class LorenzSystem(object):
    def __init__(self,
                 system_noise=0.1, observation_noise=1,
                 rho=28., s=10., beta=2.667,
                 init_range_x=(-20., 20),
                 init_range_y=(-30., 30),
                 init_range_z=(0.0, 50.),
                 init_t=2,
                 dt=0.01,
                 rng=None):
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.beta = beta + system_noise * rng.randn() * beta
        self.s = s + system_noise * rng.randn() * s
        self.rho = rho + system_noise * rng.randn() * rho
        self.dt = dt
        y_0 = [(b - a) * rng.random_sample() for a, b in (init_range_x, init_range_y, init_range_z)]
        # We run each system forwards for t_init time
        self.y = step(y_0, self.rho, self.s, self.beta, init_t, dt)

    def observe(self):
        return self.y + self.rng.randn()*self.observation_noise

    def step(self, t):
        self.y = step(self.y, self.rho, self.s, self.beta, t, self.dt)

    def generate_forecasts(self, ensemble_size, t, dt=None):
        if dt is None:
            dt = self.dt
        trajectories = []
        for i in range(ensemble_size):
            observation = self.observe()
            trajectory = solve(observation, self.rho, self.s, self.beta, t, dt)
            trajectories.append(trajectory)
        return trajectories


class DynamicDataset(object):
    """
    This class implements a \"dynamic\" version of the Lorenz system(s) instead of being backed by a pregenerated
    dataset.
    """
    def __init__(self, n_systems,
                 system_noise=0.1,
                 observation_noise=1,
                 s=10., rho=28., beta=2.667,
                 rng=None,
                 init_range_x=(-20., 20),
                 init_range_y=(-30., 30),
                 init_range_z=(0.0, 50.),
                 init_t=2,
                 dt=0.01
                 ):
        """
        Create a new dynamic Lorenz dataset.
        :param n_systems: How many different systems are there? Each system will have it's own random hyper parameters.
        :param rng: a numpy RandomState used to create the systems
        """
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng
        self.n_systems = n_systems
        self.system_noise = system_noise
        self.observation_noise = observation_noise
        self.s = s
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.systems = [LorenzSystem(system_noise, observation_noise,
                                     rho, s, beta,
                                     init_range_x, init_range_y, init_range_z,
                                     init_t, dt,
                                     rng)]


    def observe_system(self, system_id):
        return self.systems[system_id].observe()

    def step(self, system_id, t):
        self.systems[system_id].step(t)

    def step_all(self, t):
        for sys in self.systems:
            sys.step(t)

    def generate_system_forecasts(self, system_id, ensemble_size, t, dt=None):
        return self.systems[system_id].generate_forecasts(ensemble_size, t, dt)





if __name__ == '__main__':
    _calculate_canonical_statistics()