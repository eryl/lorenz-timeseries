import argparse
import numpy as np

from lorenz.lorenz import split_dataset


def split_type(x):
    name, ratio = x
    return name, float(ratio)


def main():
    parser = argparse.ArgumentParser(description="Script for splitting a Lorenz data set into multiple disjoint parts")
    parser.add_argument('datafile',
                        help="File to read data from")
    parser.add_argument('--split',
                        help="Define a split, this should be two values: a string name and a float ratio. The name will"
                             " be suffixed to the dataset base name. The ratio is the fraction of examples from the "
                             "original dataset which will be included in the split.", nargs=2, required=True,
                        action='append')
    parser.add_argument('--normalize',
                        help="If this flag is set, the split ratios will be normalized to sum to one.",
                        action='store_true')
    parser.add_argument('--random-seed', help="Value to seed the random number generator with.", type=int)
    args = parser.parse_args()
    rng = np.random.RandomState(args.random_seed)
    splits = [split_type(x) for x in args.split]
    split_dataset(args.datafile, splits, normalize=args.normalize, rng=rng)


if __name__ == '__main__':
    main()
