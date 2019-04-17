import argparse
import os.path

from lorenz.lorenz import make_dataset
DEFAULT_DATA_PATH = os.path.join(os.path.split(os.path.dirname(__file__))[0], 'data', 'lorenz.h5')
print(DEFAULT_DATA_PATH)

def main():

    parser = argparse.ArgumentParser(description="Script to generate synthetic data")
    parser.add_argument('n', help="Number of trajectories", type=int)
    parser.add_argument('--output', help="File to save the data to", default=DEFAULT_DATA_PATH)
    parser.add_argument('--data-splits', help="If given, the data for each HP setting will "
                                              "be split into files with these ratios.", type=float, nargs='+')
    parser.add_argument('-dt', help="Resolution for the ODE solver", type=float, default=0.01)
    parser.add_argument('-t', help="The time until which to run the ODE solver for", type=float, default=1000)
    parser.add_argument('--t-skip', help="Skip this much time at the start of each trajectories", type=float, default=3)
    parser.add_argument('--rng-seed', help="Constant to seed random number generator with", type=int)
    parser.add_argument('--hp-noise-level', help="If not zero is set, the hyper parameters for the Lorenz equations "
                                                 "are randomly perturbed by this amount", type=float, default=0)
    parser.add_argument('--hp-same-perturbance-ratio',
                        help="If the noise level is more than zero, this controls the fraction of samples which "
                             "receives the same hp settings. If set to 0, each trajectory uses different hyper "
                             "parameters",
                        type=float,
                        default=0)
    parser.add_argument('--hp-s', help="Value for the s hyperparameter in the Lorenz system", type=float,
                        default=10)
    parser.add_argument('--hp-beta', help="Value for the beta hyperparameter in the Lorenz system", type=float,
                        default=8/3)
    parser.add_argument('--hp-rho', help="Value for the rho hyperparameter in the Lorenz system", type=float,
                        default=28)
    args = parser.parse_args()

    make_dataset(args.output, args.n, args.t, args.dt, mode='w', t_skip=args.t_skip,
                 lorenz_beta=args.hp_beta, lorenz_s=args.hp_s, lorenz_rho=args.hp_rho,
                 noise_level=args.hp_noise_level,
                 n_ratio_same_perturbance=args.hp_same_perturbance_ratio,
                 rng_seed=args.rng_seed)






if __name__ == '__main__':
    main()