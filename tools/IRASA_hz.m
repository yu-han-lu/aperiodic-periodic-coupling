function [freqs, original_powspectra, fractal_powspectra, oscillatory_powspectra] = IRASA_hz(fs, T, signal, win_length, f_range)
% IRASA_HZ  Compute fractal and oscillatory components using IRASA (FieldTrip)
%
% Syntax:
%   [freqs, original_powspectra, fractal_powspectra, oscillatory_powspectra] = ...
%       IRASA_hz(fs, T, signal, win_length, f_range)
%
% Inputs:
%   fs          - Sampling rate (Hz)
%   T           - Duration (seconds)
%   signal      - Data matrix (channels × timepoints)
%   win_length  - Window length (seconds)
%   f_range     - Frequency range [f_min f_max] (the actual f_range)
%                 For IRASA, the actual f_range should be set to 
%                 about 1.9× wider than the frequencies you wish to observe,
%                 i.e., use [f_min/1.9, f_max*1.9].
%                 (This compensates for the resampling factors used by IRASA.)
%
% Outputs:
%   freqs                 - Frequency vector (Hz)
%   original_powspectra   - Original power spectrum (channels × freqs)
%   fractal_powspectra    - Fractal component from IRASA (channels × freqs)
%   oscillatory_powspectra- Oscillatory component (channels × freqs)
%
% Example:
%   [f, orig, frac, osc] = IRASA_hz(1000, 2, data, 1, [1 100]);
%
% Requirements:
%   - FieldTrip must be added to the MATLAB path
%   - Data will be processed channel by channel
%
% -------------------------------------------------------------------------

%% Basic setup
if nargin < 5
    error('Usage: IRASA_hz(fs, T, signal, win_length, f_range)');
end

num_chans = size(signal, 1);
num_points = T * fs;
time_axis = ((1:num_points) - 1) / fs;

% Preallocate outputs
freqs = [];
original_powspectra   = zeros(num_chans, 1);
fractal_powspectra    = zeros(num_chans, 1);
oscillatory_powspectra= zeros(num_chans, 1);

%% Loop over channels
for ch = 1:num_chans
    fprintf('Processing channel %d / %d...\n', ch, num_chans);

    % Prepare FieldTrip data structure
    data = [];
    data.trial{1} = signal(ch, :);
    data.time{1}  = time_axis;
    data.label    = {'chan'};

    % Redefine data with sliding windows
    cfg = [];
    cfg.length  = win_length;
    cfg.overlap = 0.5;
    [~, data_seg] = evalc('ft_redefinetrial(cfg, data)');

    % ----- Compute fractal and original spectra -----
    cfg = [];
    cfg.foilim = f_range;
    cfg.pad    = 'nextpow2';
    cfg.method = 'irasa';

    % Fractal
    cfg.output = 'fractal';
    [~, fractal] = evalc('ft_freqanalysis(cfg, data_seg)');

    % Original
    cfg.output = 'original';
    [~, original] = evalc('ft_freqanalysis(cfg, data_seg)');

    % ----- Compute oscillatory (original - fractal) -----
    cfg = [];
    cfg.parameter = 'powspctrm';
    cfg.operation = 'x2 - x1';
    [~, oscillatory] = evalc('ft_math(cfg, fractal, original)');

    % Save results
    if isempty(freqs)
        freqs = original.freq;
        nFreqs = numel(freqs);
        original_powspectra    = zeros(num_chans, nFreqs);
        fractal_powspectra     = zeros(num_chans, nFreqs);
        oscillatory_powspectra = zeros(num_chans, nFreqs);
    end

    original_powspectra(ch, :)    = original.powspctrm;
    fractal_powspectra(ch, :)     = fractal.powspctrm;
    oscillatory_powspectra(ch, :) = oscillatory.powspctrm;

    clear data data_seg fractal original oscillatory
end

fprintf('IRASA analysis complete.\n');

end