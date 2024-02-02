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
    parser.add_argument('datafile', type=argparse.FileType('r'))
    args = parser.parse_args()

    data = {}
    with args.datafile as f:
        for line in f:
            tokens = line.split()
            prob = tokens[0]
            case = "sk-"+tokens[1] if prob == "smoother-kernel" else prob
            tf = float(tokens[2])
            entry = np.array(list(map(float, tokens[2:])))
            data.setdefault(case, {}).setdefault(tf, []).append(entry)

    min = {}
    max = {}
    mean = {}
    med = {}

    for case in data:
        min[case] = np.array([np.min(a, 0) for a in data[case].values()])
        max[case] = np.array([np.max(a, 0) for a in data[case].values()])
        med[case] = np.array([np.median(a, 0) for a in data[case].values()])
        mean[case] = np.array([np.mean(a, 0) for a in data[case].values()])

    cycle = (
        cycler.cycler(color=['b', 'g', 'r', 'c', 'm', 'y', 'k']) +
        cycler.cycler(marker=['o', 's', 'D', '^', 'v', 'p', 'P'])
    )
    with plt.ion():
        nplots = next(iter(min.values())).shape[1]
        for i in range(2, nplots):
            plt.figure(i)
            ax = plt.gca()
            ax.cla()
            ax.set_prop_cycle(cycle)
            for case in min:
                yerr = np.array([min[case][:,i], max[case][:,i]])
                ax.loglog(
                    med[case][:,0], med[case][:,i], label=case,
                    ls='', 
                )
            #plt.gca().set_xscale('log')
            #plt.gca().set_yscale('log')
            ax.legend()
