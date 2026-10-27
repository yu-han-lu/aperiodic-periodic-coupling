#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:41:09 2026

@author: hangze
"""

import h5py
import numpy as np
from scipy.interpolate import interp1d
from scipy.io import savemat

input_path = '../data/fig1c_irasa_data.mat'
save_path = '../data/fig1d_split_exponent_result.mat'
freq_range = (20, 100)

with h5py.File(input_path, 'r') as f:
    freqs = f['f'][:]
    freqs = freqs.flatten()
    fractal_powspectra = f['fractal_powspectra'][:]

freqs_idx = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
freqs_fit = freqs[freqs_idx]
n_spectra = fractal_powspectra.shape[1]

exponent_sum = np.empty(n_spectra, dtype=object)
knee_freq_sum = np.empty(n_spectra, dtype=object)
for i in range(n_spectra):
    x = np.log10(freqs_fit)
    y = np.log10(fractal_powspectra[freqs_idx, i])

    N = 1000
    x_new = np.linspace(x.min(), x.max(), N)
    interp_func = interp1d(x, y, kind='linear', fill_value='extrapolate')
    y_new = interp_func(x_new)

    slope = np.polyfit(x_new, y_new, 1)[0]
    exponent_sum[i] = -slope
    knee_freq_sum[i] = np.nan

savemat(save_path, {
    'exponent_sum': exponent_sum,
    'knee_freq_sum': knee_freq_sum,
    'freq_range': np.array(freq_range, dtype=int),
})

print(f'split linear fit done on {n_spectra} spectra ({freq_range[0]}-{freq_range[1]} Hz)')
print(f'  exponent median={np.median(np.array(list(exponent_sum), dtype=float)):.4f}')
print(f'  saved to: {save_path}')
