import os
import functools

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # This needs to be imported for the projection='3d' to work in plot_3d
import numpy as np
from scipy import integrate
import h5py


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


def lorenz(x_y_z, t0, s=10., r=28., b=2.667):
    x, y, z = x_y_z
    x_dot = s*(y - x)
    y_dot = r*x - y - x*z
    z_dot = x*y - b*z
    return x_dot, y_dot, z_dot


def make_dataset(output, *args, mode='r+', **kwargs):
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with h5py.File(output, mode=mode) as store:
        trajectories = generate_lorenz_trajectories(*args, **kwargs)
        for y_ts, settings in trajectories:
            i = len(store.keys())
            dataset_name = 'dataset_{}'.format(i)
            dataset = store.create_dataset(dataset_name, data=y_ts)
            dataset.attrs.update(settings)

def dim_reduce_trajectories(store_file):
    with h5py.File(store_file) as store:
        datasets = []
        for dataset in store.values():
            datasets.append(dataset[:].reshape(-1, 3))
        data = np.concatenate(datasets)
        m = data.mean(axis=0)
        M = data - m
        eig_val, eig_vec = np.linalg.eigh(np.cov(M, rowvar=False))
        eig_order = np.argsort(eig_val)
        pca1 = eig_vec[:, eig_order[-1]]

        # fig = plt.figure()
        # ax = fig.gca(projection='3d')
        # np.random.seed(1)
        # data_sample_indices = np.random.choice(data.shape[0], size=1000, replace=False)
        # ax.scatter(data[data_sample_indices,0], data[data_sample_indices,1], data[data_sample_indices,2], alpha=0.5)
        # pca_line = np.zeros((3, 3), dtype=pca1.dtype)
        # pca_line[0] = m - pca1*30
        # pca_line[1] = m
        # pca_line[2] = m + pca1*30
        # ax.plot(pca_line[:,0], pca_line[:,1], pca_line[:,2], c='red')
        # plt.show()
        # return

        full_dim_datasets = [(name, dataset) for name,dataset in store.items() if dataset.shape[-1] == 3]
        for name, dataset in full_dim_datasets:
            proj_name = name + '_pca1'
            if not proj_name in store:
                pca_projection = np.dot(dataset[:], pca1)
                projection_dataset = store.create_dataset(proj_name, data=pca_projection)
                projection_dataset.attrs['pca1'] = pca1

def generate_lorenz_trajectories(n, t, dt,
                                 t_skip=0,
                                 lorenz_beta=8/3.,
                                 lorenz_s=10.,
                                 lorenz_rho=28.,
                                 noise_level=0.,
                                 n_ratio_same_perturbance=0.,
                                 rng_seed=None):
    rng = np.random.RandomState(rng_seed)

    y_ts = []

    if noise_level > 0:
        samples_per_hp_perturbance = max(int(np.floor(n_ratio_same_perturbance * n)), 1)
        num_perturbances = n // samples_per_hp_perturbance
        hp_perturbance_sizes = [samples_per_hp_perturbance] * num_perturbances
        hp_perturbance_sizes[-1] = n - sum(hp_perturbance_sizes[:-1])

        for perturbance_n in hp_perturbance_sizes:
            noisy_lorenz_beta = lorenz_beta + noise_level * rng.randn() * (8 / 3)
            noisy_lorenz_s = lorenz_s + noise_level * rng.randn() * 10
            noisy_lorenz_rho = lorenz_rho + noise_level * rng.randn() * 28
            local_rng_seed = rng.randint(2 ** 32)
            y_t = generate_trajectories(perturbance_n,
                                        t,
                                        t_skip=t_skip,
                                        dt=dt,
                                        lorenz_beta=noisy_lorenz_beta,
                                        lorenz_s=noisy_lorenz_s,
                                        lorenz_rho=noisy_lorenz_rho,
                                        rng=np.random.RandomState(local_rng_seed))
            settings = dict(beta=noisy_lorenz_beta, rho=noisy_lorenz_rho, s=noisy_lorenz_s, dt=dt, t=t,
                            t_skip=t_skip, rng_seed=local_rng_seed)

            y_ts.append((y_t, settings))
        return y_ts
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
        return [(y_t, settings)]

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


def plot_channels(y_t, plot_dims=(0,1,2)):
    num_channels, num_samples, dims = y_t.shape
    t = np.linspace(0, 1, num_samples)
    fig, axes = plt.subplots(num_channels, 1, squeeze=True)
    for ax, y_t_i in zip(axes, y_t):
        ax.plot(t, y_t_i[:,plot_dims])
    plt.show()

def plot_3d(y_t):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    for y_t_i in y_t:
        ax.plot(y_t_i[:, 0], y_t_i[:, 1], y_t_i[:, 2], lw=0.5)
    ax.set_xlabel("X Axis")
    ax.set_ylabel("Y Axis")
    ax.set_zlabel("Z Axis")
    ax.set_title("Lorenz Attractor")
    plt.show()


if __name__ == '__main__':
    main()