#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 13:41:09 2026

@author: hangze
"""

import h5py
import numpy as np
from fooof import FOOOFGroup
from scipy.io import savemat

input_path = '../data/fig1c_oscillation_welch.mat'
save_path = '../data/fig1d_fooof_result.mat'
freq_range = (0.2, 200)

with h5py.File(input_path, 'r') as f:
    f_axis = f['f'][:]
    f_axis = f_axis.flatten()
    pxx_all = f['pxx_all'][:]

fg = FOOOFGroup(aperiodic_mode='knee', max_n_peaks=1, peak_width_limits=[1.5, 6], min_peak_height=6/10, peak_threshold=2)
fg.fit(freqs=f_axis, power_spectra=pxx_all, freq_range=freq_range)

n_spectra = pxx_all.shape[0]
aperiodic = fg.get_params('aperiodic_params')
aperiodic_offset = aperiodic[:, 0]
aperiodic_knee = aperiodic[:, 1]
aperiodic_exponent = aperiodic[:, 2]

peak_cf = np.full(n_spectra, np.nan)
peak_power = np.full(n_spectra, np.nan)
peak_bandwidth = np.full(n_spectra, np.nan)
peak_table = fg.get_params('peak_params')
if peak_table.size > 0:
    peak_table = np.atleast_2d(peak_table)
    for row in peak_table:
        idx = int(row[3])
        peak_cf[idx] = row[0]
        peak_power[idx] = row[1]
        peak_bandwidth[idx] = row[2]

n_peaks = np.array(fg.n_peaks_)
r_squared = np.array([fg.get_fooof(i).r_squared_ for i in range(n_spectra)])
error = np.array([fg.get_fooof(i).error_ for i in range(n_spectra)])

savemat(save_path, {
    'f': f_axis.reshape(1, -1),
    'freq_range': np.array(freq_range, dtype=float).reshape(1, -1),
    'aperiodic_offset': aperiodic_offset.reshape(1, -1),
    'aperiodic_knee': aperiodic_knee.reshape(1, -1),
    'aperiodic_exponent': aperiodic_exponent.reshape(1, -1),
    'peak_cf': peak_cf.reshape(1, -1),
    'peak_power': peak_power.reshape(1, -1),
    'peak_bandwidth': peak_bandwidth.reshape(1, -1),
    'n_peaks': n_peaks.reshape(1, -1),
    'r_squared': r_squared.reshape(1, -1),
    'error': error.reshape(1, -1),
})

n_detected = int(np.sum(n_peaks > 0))
print(f'FOOOF fitting done on {n_spectra} spectra')
print(f'  detected peaks: {n_detected}/{n_spectra}')
print(f'  saved to: {save_path}')
