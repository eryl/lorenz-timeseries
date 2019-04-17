import argparse

from lorenz.lorenz import dim_reduce_trajectories

def main():
    parser = argparse.ArgumentParser(description="Use PCA to reduce the 3-dimensional time series to a 1D")
    parser.add_argument('datafile', help="File from which to calculate PCA and add 1D projections to")
    args = parser.parse_args()
    dim_reduce_trajectories(args.datafile)




if __name__ == '__main__':
    main()
