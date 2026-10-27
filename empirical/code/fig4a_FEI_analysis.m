%%
disp(['preprocessing file: ', matfile]);
addpath('../../tools/');

% sub-xx
[~, name, ~] = fileparts(matfile);
nums = str2double(regexp(name, '\d+', 'match'));

%%
% input data
data = load(matfile);
fs = data.fs;
data_sum = data.data;
clear data

% envelope
yb_envelope = abs(hilbert(data_sum));

% segment
win_len = 5*fs;
step_len = win_len/2;
seg_num = floor((size(yb_envelope,1)-win_len)./step_len)+1;

yb_seg = [];
for i = 1:seg_num
    start_idx = (i-1)*step_len+1;
    end_idx = start_idx+win_len-1;
    yb_seg(i, :, :) = yb_envelope(start_idx:end_idx, :);
    clear start_idx end_idx
end
clear win_len step_len seg_num
clear yb_envelope

% calculate FEI
EI_sum = [];
for i = 1:size(yb_seg, 1)
    [EI_sum(i, :), ~, ~] = calculateFEI(squeeze(yb_seg(i, :, :)), fs, 0.8);
end

% save data
save_path = '../data/FEI_data/';
save_name = [name '.mat'];
save(fullfile(save_path, save_name), 'EI_sum', '-v7.3');