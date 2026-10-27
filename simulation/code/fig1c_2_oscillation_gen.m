%% Add band-limited oscillations in time domain, Welch PSD
clear; clc; close all; warning off;

script_dir = '../data/';
ou_path = fullfile(script_dir, 'fig1c_OU_data.mat');
out_pxx = fullfile(script_dir, 'fig1c_oscillation_welch.mat');
out_x   = fullfile(script_dir, 'fig1c_oscillation_timedomain.mat');

n_spectra = 100;
cf_range = [30, 150];
bw_range = [1.5, 6.0];
power_frac_range = [2.8, 3.8];
fs = 1000;

rng(42);

S = load(ou_path);
x_base = S.x(:)';
window = 30 * fs;
win = hamming(window, 'periodic');
noverlap = round(window / 2);
nfft = 2^nextpow2(window);

[pxx, f] = pwelch(x_base, win, noverlap, nfft, fs, 'onesided');
pxx = pxx(:);
f = f(:);

osc_params = struct( ...
    'center_freq', cell(n_spectra, 1), ...
    'power',       cell(n_spectra, 1), ...
    'power_frac',  cell(n_spectra, 1), ...
    'bandwidth',   cell(n_spectra, 1), ...
    'amplitude',   cell(n_spectra, 1));

pxx_all = zeros(numel(f), n_spectra);
x_all = zeros(n_spectra, numel(x_base));

for k = 1:n_spectra
    cf = cf_range(1) + (cf_range(2) - cf_range(1)) * rand();
    pf = power_frac_range(1) + (power_frac_range(2) - power_frac_range(1)) * rand();
    bw = bw_range(1) + (bw_range(2) - bw_range(1)) * rand();

    osc_unit = make_band_osc(x_base, fs, cf, bw);
    amp = calibrate_amplitude(x_base, osc_unit, f, pxx, cf, pf, win, noverlap, nfft, fs);

    x_new = x_base + amp * osc_unit;
    [pxx_new, ~] = pwelch(x_new, win, noverlap, nfft, fs, 'onesided');

    local_base = interp1(f, pxx, cf, 'linear', 'extrap');
    bump = interp1(f, pxx_new, cf, 'linear', 'extrap') - local_base;

    osc_params(k).center_freq = cf;
    osc_params(k).power = bump;
    osc_params(k).power_frac = bump / local_base;
    osc_params(k).bandwidth = bw;
    osc_params(k).amplitude = amp;
    x_all(k, :) = x_new;
    pxx_all(:, k) = pxx_new(:);
end

save(out_x, 'x_base', 'x_all', 'fs', 'osc_params', '-v7.3');
save(out_pxx, 'f', 'pxx', 'pxx_all', 'osc_params', '-v7.3');
fprintf('Saved %s\n', out_x);
fprintf('Saved %s\n', out_pxx);


function osc = make_band_osc(x, fs, cf, bw)
    wn = randn(size(x));
    low = max((cf - bw / 2) / (fs / 2), 0.001);
    high = min((cf + bw / 2) / (fs / 2), 0.999);
    [b, a] = butter(2, [low, high], 'bandpass');
    osc = filtfilt(b, a, wn);
    osc = osc / rms(osc);
end


function amp = calibrate_amplitude(x, osc_unit, f, pxx_base, cf, target_frac, win, noverlap, nfft, fs)
    target_bump = target_frac * interp1(f, pxx_base, cf, 'linear', 'extrap');

    [pxx_unit, ~] = pwelch(x + osc_unit, win, noverlap, nfft, fs, 'onesided');
    bump_unit = interp1(f, pxx_unit, cf, 'linear', 'extrap') - interp1(f, pxx_base, cf, 'linear', 'extrap');

    if bump_unit <= 0
        amp = 1.0;
        return;
    end

    amp = sqrt(max(target_bump, eps) / bump_unit);
end
