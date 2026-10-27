%%
clear; clc; close all; warning off;

%%
sublist = [22 28 65 68 69 70 71 72 73 74 75 76 77 78 85 89 92 96 97 98 117 123 131 135 140 141 144 152 153 159 164 ...
    168 169 176 177 186 189 190 192 197 198 201 204 205 213 216 229 241 245 246 253 255 256 257 258 259 260 261 262 263 264];

% loop-subject
rho_sum_subj = [];
pval_sum_subj = [];
for idx_subj = sublist

    % loop-channel
    rho_sum = [];
    pval_sum = [];
    input_path = sprintf('../data/spilt_data/sub-%d/', idx_subj);
    matFiles = dir(fullfile(input_path, '*.mat'));
    channel_num = length(matFiles)./5;
    for j = 1:channel_num
        try
            % loop-window
            for i = 1:5
                input_path = sprintf('../data/spilt_data/sub-%d/sub-%d-%d-%d.mat', idx_subj, idx_subj, j, i);
                data = load(input_path);
                knee_sum = data.knee_freq_sum_avg';
                exponent_sum = data.exponent_sum_avg';
                clear data

                input_path = sprintf('../data/oscillation_data/sub-%d/sub-%d-%d-%d.mat', idx_subj, idx_subj, j, i);
                data = load(input_path);
                peak_sum = data.peak_sum';
                clear data

                % loop-average
                for idx = 1:length(knee_sum)

                    % single result
                    knee_avg = knee_sum{idx}';
                    exponent_avg = exponent_sum{idx}';
                    peak_avg = peak_sum(:, idx);

                    if isa(exponent_avg, 'double')
                        exponent_avg = num2cell(exponent_avg, 1)';
                    end

                    % peak
                    peak_freq = nan(size(knee_avg));
                    peak_power = nan(size(knee_avg));
                    peak_exponent = nan(size(knee_avg));
                    for k = 1:size(knee_avg, 1)
                        temp_peak = peak_avg{k, 1};
                        temp_exponent = exponent_avg{k, 1};
                        if any(temp_peak ~= 0, 'all')
                            if length(temp_exponent) == 2 && temp_exponent(1) < temp_exponent(2)
                                [~, temp_idx] = max(temp_peak(:, 2));
                                peak_freq(k, 1) = temp_peak(temp_idx, 1);
                                peak_power(k, 1) = temp_peak(temp_idx, 2);
                                peak_exponent(k, 1) = temp_exponent(2);
                            end
                            clear temp_idx
                        end
                        clear temp_*
                    end
                    clear knee_avg exponent_avg peak_avg

                    % correlation
                    rho_band = nan(5, 1);
                    pval_band = nan(5, 1);
                    temp_exponent = peak_exponent(~isnan(peak_exponent));
                    temp_frequency = peak_freq(~isnan(peak_freq));
                    clear peak_power peak_freq peak_exponent

                    % band
                    theta = temp_frequency >= 1  & temp_frequency < 8;
                    alpha = temp_frequency >= 8  & temp_frequency < 13;
                    beta = temp_frequency >= 13 & temp_frequency < 30;
                    lowgamma = temp_frequency >= 30 & temp_frequency < 80;
                    highgamma = temp_frequency >= 80 & temp_frequency < 150;
                    bands = {theta, alpha, beta, lowgamma, highgamma};

                    for l = 1:length(bands)
                        temp_idx = bands{l};
                        temp_exponent_band = temp_exponent(temp_idx);
                        temp_frequency_band = temp_frequency(temp_idx);
                        temp_exponent_clean = remove_outlier(temp_exponent_band);
                        temp_frequency_clean = remove_outlier(temp_frequency_band);
                        temp_nan_idx = find(~isnan(temp_exponent_clean) & ~isnan(temp_frequency_clean));
                        if length(temp_nan_idx) >= 20
                            [rho, pval] = corr(temp_exponent_clean, temp_frequency_clean, ...
                                'Type', 'Spearman', 'Rows', 'pairwise');
                            rho_band(l) = rho;
                            pval_band(l) = pval;
                            clear rho pval
                        end
                        clear temp_idx temp_exponent_* temp_frequency_*
                    end
                    rho_sum(j, i, idx, :) = rho_band;
                    pval_sum(j, i, idx, :) = pval_band;
                    clear temp_* rho_band pval_band
                end
            end
        catch
            rho_sum(j, i, idx, :) = nan(5, 1);
            pval_sum(j, i, idx, :) = nan(5, 1);
            disp([num2str(idx_subj) ' and ' num2str(j)]);
        end
    end
    rho_sum_subj = [rho_sum_subj; rho_sum];
    pval_sum_subj = [pval_sum_subj; pval_sum];
    clear rho_sum pval_sum
end

% save data
save_path = '../data/fig1e_split_frequency_coupling.mat';
save(save_path, 'rho_sum_subj', 'pval_sum_subj');

function x_clean = remove_outlier(x)

Q1 = prctile(x, 25);
Q3 = prctile(x, 75);
IQR = Q3 - Q1;

lower_bound = Q1 - 1.5 * IQR;
upper_bound = Q3 + 1.5 * IQR;

mask  = x >= lower_bound & x <= upper_bound;
x_clean = x;
x_clean(~mask) = nan;

end