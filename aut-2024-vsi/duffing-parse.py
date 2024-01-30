#!/usr/bin/env python3

import argparse
import pathlib
import pickle

import numpy as np


def parse(dir):
    cases = filter(pathlib.Path.is_dir, pathlib.Path(dir).glob('*'))
    data = {}
    for case in cases:
        data[case.name] = []
        with open( case.name + ".txt", "w") as outfile:
            entries = case.glob('tf_*')
            for entry in entries:
                tf = float(entry.name[3:])
                time = []
                qerr = []
                xerr = []
                no_result = []
                for rep in entry.glob('*.pickle'):
                    with open(rep, 'rb') as f:
                        try:
                            data = pickle.load(f)
                            xsim = data['sim']['x'][::4]
                            if case.name.startswith('sk-'):
                                skip = data['args']['nwin'] // 2
                                xsim = xsim[skip:-skip]
                            qsim = data['sim']['q']
                            qest = data['decopt']['q']
                            xest = data['vopt']['xbar']
                            qerr.append(np.abs(qsim - qest))
                            xerr.append(np.abs(xsim - xest).mean(0))
                            time.append(abs(data['time_elapsed']))
                        except Exception as e:
                            no_result.append(rep)
                            print(rep, e)
                data[case.name].append(
                    tf=tf,
                    time=time,
                    qerr=qerr,
                    xerr=xerr,
                    no_result=no_result,
                )
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=pathlib.Path, default=pathlib.Path('.'))
    args = parser.parse_args()

    data = parse(args.dir)

    # Write binary data
    with open(args.dir / 'data.pickle', 'wb') as f:
        pickle.dump(data, f)
    
    # Write text data    
    for k, v in data.items():
        with open(args.dir / (k + '.txt'), 'w') as f:
            for entry in v:
                f.write(
                    entry.tf,
                    entry.time.min(), entry.time.mean(), entry.time.max(),
                    *entry.qerr.min(0), *entry.qerr.mean(0), *entry.qerr.max(0),
                    *entry.xerr.min(0), *entry.xerr.mean(0), *entry.xerr.max(0),
                )
