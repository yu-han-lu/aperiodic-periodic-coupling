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

        % welch
        window = window_sum(i)*fs;
        win = hamming(window, 'periodic');
        noverlap = round(window ./ 2);
        nfft = 2^nextpow2(window);

        pxx = [];
        f = [];
        for j = 1:size(yb_seg, 1)
            x = squeeze(yb_seg(j, :, :));
            [pxx(j, :, :), f] = pwelch(x, win, noverlap, nfft, fs, 'onesided');
            clear x
        end
        clear yb_seg

        f_idx = find(f >= 0.2 & f <= 200);
        f = f(f_idx);
        pxx = pxx(:, f_idx, :);
        clear f_idx

        % loop-avg
        avg_num = [2 4 6 8 10];
        pxx_avg = {};
        for j = 1:length(avg_num)
            pxx_avg{j} = movmean(pxx, avg_num(j), 1, 'Endpoints', 'discard');
        end
        clear pxx

        % save data
        save_name = sprintf('sub-%d-%d.mat', nums(1), i);
        save_path = '../data/welch_data/';
        save(fullfile(save_path, save_name), 'pxx_avg', 'f', '-v7.3');
        clear f pxx_avg
    end
end