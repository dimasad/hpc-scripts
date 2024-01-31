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
                            res = pickle.load(f)
                            xsim = res['sim']['x'][::4]
                            if case.name.startswith('sk-'):
                                skip = res['args']['nwin'] // 2
                                xsim = xsim[skip:-skip]
                            qsim = res['sim']['q']
                            qest = res['decopt']['q']
                            xest = res['vopt']['xbar']
                            qerr.append(np.abs(qsim - qest))
                            xerr.append(np.abs(xsim - xest).mean(0))
                            time.append(abs(res['time_elapsed']))
                        except Exception as e:
                            no_result.append(rep)
                            print(rep, e)
                data[case.name].append(
                    dict(
                        tf=tf,
                        time=time,
                        qerr=qerr,
                        xerr=xerr,
                        no_result=no_result,
                    )
                )
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=pathlib.Path, default=pathlib.Path('.'))
    parser.add_argument('--prefix', type=str, default="")
    args = parser.parse_args()

    data = parse(args.dir)

    # Write binary data
    with open(args.dir / (args.prefix + 'pickle'), 'wb') as f:
        pickle.dump(data, f)
    
    # Write text data    
    for k, v in data.items():
        with open(args.dir / (args.prefix + k + '.txt'), 'w') as f:
            print(
                "tf t_min t_med t_max "
                "a_min b_min d_min g_min logstdw_min logstdv_min " 
                "a_med b_med d_med g_med logstdw_med logstdv_med "
                "a_max b_max d_max g_max logstdw_max logstdv_max "
                "x0_min x1_min x0_med x1_med x0_max x1_max", file=f
            )

            for entry in v:
                time = np.array(entry['time'])
                xerr = np.array(entry['xerr'])
                qerr = np.array(entry['qerr'])
                
                if time.size == 0:
                    continue

                print(
                    entry['tf'],
                    time.min(), time.mean(), time.max(),
                    *qerr.min(0), *qerr.mean(0), *qerr.max(0),
                    *xerr.min(0), *xerr.mean(0), *xerr.max(0),
                    file=f
                )
