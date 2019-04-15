import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import functools

import numpy as np
from scipy import integrate


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


def generate_lorenz_trajectories(n, t, dt,
                                 t_skip=0,
                                 lorenz_beta=8/3.,
                                 lorenz_s=10.,
                                 lorenz_rho=28.,
                                 noise_level=0.,
                                 n_ratio_same_perturbance=0.,
                                 rng=None):
    if rng is None:
        rng = np.random.RandomState()

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
            y_t = generate_trajectories(perturbance_n,
                                        t,
                                        t_skip=t_skip,
                                        dt=dt,
                                        lorenz_beta=noisy_lorenz_beta,
                                        lorenz_s=noisy_lorenz_s,
                                        lorenz_rho=noisy_lorenz_rho,
                                        rng=rng)
            settings = dict(beta=lorenz_beta, rho=lorenz_rho, s=lorenz_s)

            y_ts.append((y_t, settings))
        return y_ts
    else:
        y_t = generate_trajectories(n,
                                    t,
                                    t_skip=t_skip,
                                    dt=dt,
                                    lorenz_beta=lorenz_beta,
                                    lorenz_s=lorenz_s,
                                    lorenz_rho=lorenz_rho,
                                    rng=rng)
        return y_t

def main():
    rng = np.random.RandomState(1729)
    dt = 0.01
    n = 100  # Numbers of trajectories
    t_max = 10
    t_skip = 1
    lorenz_beta = 8 / 3
    lorenz_s = 10
    lorenz_rho = 28
    noise_level = 0.1

    y_t = generate_lorenz_trajectories(n, t_max, dt, t_skip=t_skip, lorenz_beta=lorenz_beta, lorenz_s=lorenz_s,
                                       lorenz_rho=lorenz_rho, noise_level=noise_level, rng=rng)
    plot_3d(y_t)

    #plot_channels(t, y_t)


def plot_channels(t, y_t):
    num_channels = y_t.shape[0]
    fig, axes = plt.subplots(num_channels, 1, squeeze=True)
    for ax, y_t_i in zip(axes, y_t):
        ax.plot(t, y_t_i)
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