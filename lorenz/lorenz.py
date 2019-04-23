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
            group_name = 'dataset_{}'.format(i)
            group = store.create_group(group_name)
            group.attrs.update(settings)
            dataset_3d = group.create_dataset('lorenz3d', data=y_ts)
            y_ts_1d, pc1 = dim_reduce_trajectories(y_ts)
            dataset_1d = group.create_dataset('lorenz_pca1d', data=y_ts_1d)
            dataset_1d.attrs['pc1'] = pc1


def dim_reduce_trajectories(data, n_components=1):
    """
    Reduce the dimensionality of the trajectories data using PCA.
    :param data: A 3d array with shape (num_trajectories, n, p), where n is the number of samples in the trajectory
                 and p number of features
    :param n_components: The number of principal components to use,
    :return: A 3d array of shape (num_trajectories, n, n_components)
    """
    num_trajectories, n, p = data.shape
    all_data = data.reshape((num_trajectories*n, p))
    m = all_data.mean(axis=0)
    centered_data = all_data - m
    eig_val, eig_vec = np.linalg.eigh(np.cov(centered_data, rowvar=False))
    eig_order = np.argsort(eig_val)[::-1]
    principal_components = eig_vec[:, eig_order[:n_components]]  # The principal components are column vectors

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


def generate_lorenz_trajectories(n, t, dt,
                                 t_skip=0,
                                 lorenz_beta=8/3.,
                                 lorenz_s=10.,
                                 lorenz_rho=28.,
                                 noise_level=0.,
                                 n_per_perturbance=0.,
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
    plt.show()


if __name__ == '__main__':
    main()