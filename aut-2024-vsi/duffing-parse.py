#!/usr/bin/env python3

import pathlib
import pickle

import numpy as np


if __name__ == '__main__':
    cases = filter(pathlib.Path.is_dir, pathlib.Path('.').glob('*'))
    for case in cases:
        with open( case.name + ".txt", "w") as outfile:
            entries = case.glob('tf_*')
            for entry in entries:
                tf = float(entry.name[3:])
                time = []
                qerr = []
                qsyserr = []
                x0err = []
                x1err = []
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
                            qerr.append(np.abs(qsim - qest).mean())
                            qsyserr.append(np.abs(qsim - qest)[:-2].mean())
                            x0err.append(np.abs(xsim[:,0] - xest[:,0]).mean())
                            x1err.append(np.abs(xsim[:,1] - xest[:,1]).mean())
                            time.append(abs(data['time_elapsed']))
                        except Exception as e:
                            print(rep, e)
                    
                    print(
                        tf,
                        np.min(time), np.mean(time), np.max(time), 
                        np.min(qerr), np.mean(qerr), np.max(qerr),
                        np.min(qsyserr), np.mean(qsyserr), np.max(qsyserr),
                        np.min(x0err), np.mean(x0err), np.max(x0err),
                        np.min(x1err), np.mean(x1err), np.max(x1err),
                        file=outfile
                    )