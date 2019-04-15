import argparse
import h5py

from vindel.lorenz import plot_3d, plot_channels

def main():
    parser = argparse.ArgumentParser(description="Script to generate synthetic data")
    parser.add_argument('datafile', help="File to read data from")
    parser.add_argument('--plot-dims', help="The dimensions to plot", default=[0,1,2])
    parser.add_argument('--plot-style', help="How to plot, 2d or 3d", choices=('2d', '3d'),
                        default='3d')
    args = parser.parse_args()
    with h5py.File(args.datafile) as store:
        for dataset_name, dataset in store.items():
            if args.plot_style == '3d':
                plot_3d(dataset)
            elif args.plot_style == '2d':
                plot_channels(dataset, plot_dims=args.plot_dims)




if __name__ == '__main__':
    main()
