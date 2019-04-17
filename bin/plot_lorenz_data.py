import argparse
import h5py

from lorenz.lorenz import plot_3d, plot_channels

def main():
    parser = argparse.ArgumentParser(description="Script to plot lorenz data.")
    parser.add_argument('datafile',
                        help="File to read data from")
    parser.add_argument('--plot-dims',
                        help="The dimensions to plot", nargs='*', type=int, default=[0,1,2])
    parser.add_argument('--plot-style',
                        help="How to plot, 2d or 3d", choices=('2d', '3d'),
                        default='3d')
    parser.add_argument('--plot-projection', help="Also plot the 1D projected data (if available)", action='store_true')
    args = parser.parse_args()
    with h5py.File(args.datafile) as store:
        for dataset_name, group in store.items():
            dataset = group['lorenz3d'][:]
            pca_projection = None
            pc1 = None
            if args.plot_projection and 'lorenz_pca1d' in group:
                pca_projection = group['lorenz_pca1d'][:]
                pc1 = group['lorenz_pca1d'].attrs['pc1']
            if args.plot_style == '3d':
                plot_3d(dataset, pca_projection=pca_projection, pc1=pc1)
            elif args.plot_style == '2d':
                plot_channels(dataset, plot_dims=args.plot_dims, pca_projection=pca_projection)





if __name__ == '__main__':
    main()
