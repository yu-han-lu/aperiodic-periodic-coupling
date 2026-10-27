%%
clear; clc; close all; warning off;

%%
input_path = '../data/preproc_data';
matFiles = dir(fullfile(input_path, '*.mat'));

% loop-subject
for sub = 1:length(matFiles)
    filename = matFiles(sub).name;
    nums = str2double(regexp(name, '\d+', 'match'));

    data = load(fullfile(input_path, filename));
    fs = data.fs;
    yb = data.data(1:30*60*fs, :);
    clear data

    % loop-window
    window_sum = [1 2 5 10 30];
    for i = 1:length(window_sum)
        win_len = window_sum(i)*fs;
        step_len = win_len/2;
        seg_num = floor((size(yb,1)-win_len)./step_len)+1;

        yb_seg = [];
        for j = 1:seg_num
            start_idx = (j-1)*step_len+1;
            end_idx = start_idx+win_len-1;
            yb_seg(j, :, :) = yb(start_idx:end_idx, :);
            clear start_idx end_idx
        end
        clear win_len step_len seg_num

        % irasa
        T = size(yb_seg, 2)./fs;
        win_length = window_sum(i);
        f_range = [0.01 395];
        for j = 1:size(yb_seg, 1)
            [f, original_powspectra(j, :, :), fractal_powspectra(j, :, :), oscillatory_powspectra(j, :, :)] = ...
                IRASA_hz(fs, T, squeeze(yb_seg(j, :, :))', win_length, f_range);
        end

        % loop-average
        avg_num = [2 4 6 8 10];
        pxx_avg = {};
        for j = 1:length(avg_num)
            original_powspectra_avg = movmean(original_powspectra, avg_num(j), 1, 'Endpoints', 'discard');
            fractal_powspectra_avg = movmean(fractal_powspectra, avg_num(j), 1, 'Endpoints', 'discard');
            oscillatory_powspectra_avg = movmean(oscillatory_powspectra, avg_num(j), 1, 'Endpoints', 'discard');
            
            % save data
            save_path = '../data/irasa_data';
            save_name = sprintf('sub-%d-%d-%d', nums(1), i, j);
            save(fullfile(save_path, save_name), 'original_powspectra_avg', 'fractal_powspectra_avg', 'oscillatory_powspectra_avg', 'f', '-v7.3');
            clear original_powspectra_avg fractal_powspectra_avg oscillatory_powspectra_avg
        end
    end
end