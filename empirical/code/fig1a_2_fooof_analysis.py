# -*- coding: utf-8 -*-
"""
Created on Sat May 10 22:35:48 2025

@author: 86132
"""

from fooof.utils import interpolate_spectrum
import numpy as np
from fooof import FOOOFGroup
import os
import h5py
from multiprocessing import Pool
from scipy.io import savemat

input_path = '../data/welch_data/'
output_path = '../data/fooof_data/'

def convert_knee_val(knee, exponent=2.):
    """
    Convert knee parameter to frequency and time-constant value.
    Can operate on array or float.

    Default exponent value of 2 means take the square-root, but simulation shows
    taking the exp-th root returns a more accurate drop-off frequency estimate
    when the PSD is actually Lorentzian.
    """
    knee_freq = knee**(1./exponent)
    knee_tau = 1./(2*np.pi*knee_freq)
    return knee_freq, knee_tau


def read_cell_dataset(f, varname):
    refs = f[varname][:]
    result = []

    for idx in np.ndindex(refs.shape):
        ref = refs[idx]
        item = np.array(f[ref][()]).squeeze()
        result.append(item)

    return result

def process_file(filename):
    # input mat data
    file_path = os.path.join(input_path, filename)
    with h5py.File(file_path, 'r') as f:
        f_axis = f['f'][:]
        f_axis = f_axis.flatten()
        pxx_avg = read_cell_dataset(f, 'pxx_avg')
        
    # mkdir
    name = os.path.splitext(filename)[0]
    parts = name.split('-')
    name_part = f"{parts[0]}-{parts[1]}"
    save_path = os.path.join(output_path, name_part)
    os.makedirs(save_path, exist_ok=True)
    
    # fit with fooof
    for i in range(len(pxx_avg)):
        try:
            pxx_one = pxx_avg[i]
            
            fit_exp_sum = []
            fit_offset_sum = []
            fit_knee_sum = []
            peak_params_sum = []
            for j in range(pxx_one.shape[0]):
                pxx_channel = pxx_one[j, :, :].T
                
                # deal with line noise
                spec_int_all = []
                interp_ranges = [[47, 53], [97, 103], [147, 153]]
                for k in range(pxx_channel.shape[0]):
                    freqs_int, spec_int = interpolate_spectrum(
                        f_axis, pxx_channel[k, :], interp_ranges)
                    spec_int_all.append(spec_int)
                spec_int_array = np.array(spec_int_all)
        
                fg = FOOOFGroup(aperiodic_mode='knee', max_n_peaks=3, peak_width_limits=[
                                1.5, 6], min_peak_height=6/10, peak_threshold=2)
                fg.fit(freqs=f_axis, power_spectra=spec_int_array, freq_range=(0.2, 195))
        
                # aperiodic
                fit_knee = fg.get_params('aperiodic_params', 'knee')
                fit_exp = fg.get_params('aperiodic_params', 'exponent')
                fit_offset = fg.get_params('aperiodic_params', 'offset')
                knee_freq, taus = convert_knee_val(fit_knee, fit_exp)
        
                fit_exp_sum.append(fit_exp)
                fit_offset_sum.append(fit_offset)
                fit_knee_sum.append(knee_freq)
        
                # oscillation
                peak_params = []
                for k in range(pxx_channel.shape[0]):
                    fm = fg.get_fooof(ind=k, regenerate=True)
                    peak_params.append(fm.peak_params_)
                peak_params_sum.append(peak_params)
    
            fit_exp_sum = np.array(fit_exp_sum, dtype=object)
            fit_offset_sum = np.array(fit_offset_sum, dtype=object)
            fit_knee_sum = np.array(fit_knee_sum, dtype=object)
            peak_params_sum = np.array(peak_params_sum, dtype=object)
    
            # save data
            save_name = f"{parts[0]}-{parts[1]}-{parts[2]}-{i+1}.mat"
            savemat(os.path.join(save_path, save_name), {'fit_exp_sum': fit_exp_sum, 'fit_offset_sum': fit_offset_sum, 
                                                         'fit_knee_sum': fit_knee_sum, 'peak_params_sum': peak_params_sum})

        except Exception as e:
            print(e)
            
if __name__ == '__main__':
    all_files = [f for f in os.listdir(input_path) if f.endswith('.mat')]
    num_workers = 10

    with Pool(processes=num_workers) as pool:
        results = pool.map(process_file, all_files)
