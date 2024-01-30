#!/usr/bin/env python3


import argparse
import pathlib

import numpy as np
import cycler
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--loglog', action='store_true')
    parser.add_argument('--x', type=int, default=0)
    parser.add_argument('--y', type=int, nargs='*', default=[8])
    args = parser.parse_args()

    datafiles = pathlib.Path('.').glob('*.txt')

    with plt.ion():
        fig, ax = plt.subplots()
        for datafile in datafiles:
            data = np.loadtxt(datafile)
            if args.loglog:
                ax.loglog(
                    data[:,args.x], data[:,args.y], '.', label=datafile.name
                )
            else:
                ax.plot(
                    data[:,args.x], data[:,args.y], '.', label=datafile.name
                )
        ax.legend()
