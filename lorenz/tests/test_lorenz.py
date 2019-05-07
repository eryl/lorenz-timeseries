import unittest
import numpy as np
import matplotlib.pyplot as plt

import lorenz.lorenz
from lorenz.lorenz import plot_3d

class TestDynamicDataset(unittest.TestCase):
    def test_random_iterator_1d(self):
        rng = np.random.RandomState(1729)
        n_systems = 1
        n_forecasts = 10
        t = 0.2
        lead_time = 1.2
        ensemble_size = 20
        dataset = lorenz.lorenz.DynamicDataset(n_systems, rng=rng)
        # Issue a forecast every 1 timesteps, of length 6
        observations = [[] for i in range(n_systems)]
        forecasts = [[] for i in range(n_systems)]
        for i in range(n_forecasts):
            for system_id in range(n_systems):
                system_observations = dataset.observe_system(system_id)
                system_forecasts = dataset.generate_system_forecasts(system_id, ensemble_size, lead_time)
                observations[system_id].append(system_observations)
                forecasts[system_id].append(system_forecasts)
                dataset.step(system_id, t)
        for system_id in range(n_systems):
            for t in range(n_forecasts):
                b = np.array(forecasts[system_id][t])
                plot_3d(b)
                plt.show()





