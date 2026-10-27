# -*- coding: utf-8 -*-

from scipy.io import loadmat
import numpy as np
import pwlf
from GPyOpt.methods import BayesianOptimization
from scipy.interpolate import interp1d
import os
import matplotlib
matplotlib.use('Agg')
import argparse
from scipy.io import savemat

parser = argparse.ArgumentParser()
parser.add_argument('--matfile', type=str, required=True)
parser.add_argument('--outfile', type=str, required=True)
args = parser.parse_args()

matfile = args.matfile
outfile = args.outfile

print(f'Processing: {matfile}')
print(f'Output: {outfile}')

data = loadmat(matfile)

freqs = data['freqs']
freqs = freqs.flatten()

fractal_powspectra_avg = data['fractal_powspectra_avg']
total_avg = fractal_powspectra_avg.shape[1]

freqs_idx = np.where((freqs >= 0.2) & (freqs <= 195))[0]
freqs = freqs[freqs_idx]

knee_freq_sum_avg = []
exponent_sum_avg = []

for i in range(fractal_powspectra_avg.shape[1]):

    fractal_powspectra_one = fractal_powspectra_avg[0, i]
    fractal_powspectra_one = fractal_powspectra_one[:, freqs_idx]
    total_time = fractal_powspectra_one.shape[0]

    knee_freq_sum = []
    exponent_sum = []

    for idx, h in enumerate(range(fractal_powspectra_one.shape[0])):

        if idx % 20 == 0:
            print(f'AVG {i+1}/{total_avg} | TIME {idx+1}/{total_time}', flush=True)

        try:
            x = np.log10(freqs)
            y = np.log10(fractal_powspectra_one[h, :])

            # remove NaN / Inf
            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            N = 1000
            x_new = np.linspace(x.min(), x.max(), N)
            interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')
            y_new = interp_func(x_new)

            my_pwlf = pwlf.PiecewiseLinFit(x_new, y_new)

            l = 0.5

            def my_obj(xopt):
                f = np.zeros(xopt.shape[0])

                for ii, jj in enumerate(xopt):
                    try:
                        my_pwlf.fit(int(jj[0]))
                        f[ii] = my_pwlf.ssr + (l * jj[0] * jj[0])
                    except Exception:
                        f[ii] = 1e10

                return f

            bounds = [{'name': 'n_segments', 'type': 'discrete', 'domain': np.arange(1, 5)}]
            np.random.seed(42)

            myBopt = BayesianOptimization(
                my_obj,
                domain=bounds,
                model_type='GP',
                initial_design_numdata=20,
                initial_design_type='latin',
                exact_feval=True,
                verbosity=True,
                verbosity_model=False
            )

            max_iter = 30
            myBopt.run_optimization(max_iter=max_iter, verbosity=False)

            opt_segments = int(myBopt.x_opt[0])
            breakpoints = my_pwlf.fit(opt_segments)

            exponent = -my_pwlf.slopes

            if len(exponent) == 0:
                exponent_sum.append(np.nan)
            else:
                exponent_sum.append(exponent)

            knee_freq = breakpoints[1:-1]

            if len(knee_freq) == 0:
                knee_freq_sum.append(np.nan)
            else:
                knee_freq_log = 10 ** np.array(knee_freq)
                knee_freq_sum.append(knee_freq_log)

        except Exception as e:
            print(f'ERROR | AVG {i+1}/{total_avg} | TIME {idx+1}/{total_time} | {e}', flush=True)

            knee_freq_sum.append(np.nan)
            exponent_sum.append(np.nan)

            continue

    knee_freq_sum_avg.append(knee_freq_sum)
    exponent_sum_avg.append(exponent_sum)

knee_freq_sum_avg = np.array(knee_freq_sum_avg, dtype=object)
exponent_sum_avg = np.array(exponent_sum_avg, dtype=object)

os.makedirs(os.path.dirname(outfile), exist_ok=True)

savemat(outfile, {
    'knee_freq_sum_avg': knee_freq_sum_avg,
    'exponent_sum_avg': exponent_sum_avg
})