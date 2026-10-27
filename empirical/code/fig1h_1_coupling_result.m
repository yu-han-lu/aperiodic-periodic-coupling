%%
clear; clc; close all; warning off;

%%
sublist = [22 28 65 68 69 70 71 72 73 74 75 76 77 78 85 89 92 96 97 98 117 123 131 135 140 141 144 152 153 159 164 ...
    168 169 176 177 186 189 190 192 197 198 201 204 205 213 216 229 241 245 246 253 255 256 257 258 259 260 261 262 263 264];

% loop-subject
rho_sum_subj = [];
pval_sum_subj = [];
for idx_subj = sublist
    input_path = sprintf('../data/split_data/sub-%d.mat', idx_subj);
    data = load(input_path);
    exponent_subj = data.exponent_subj;
    clear data

    input_path = sprintf('../data/fooof_data/sub-%d.mat', idx_subj);
    data = load(input_path);
    peak_subj = data.peak_subj;
    clear data

    % judge
    rho_sum = [];
    pval_sum = [];
    if size(exponent_subj, 1) == size(peak_subj, 1)
        % loop-channel
        for i = 1:size(exponent_subj, 1)
            % loop-window
            for j = 1:size(exponent_subj, 2)
                % loop-average
                for k = 1:size(exponent_subj, 3)
                    exponent_one = exponent_subj{i, j, k}';
                    peak_one = peak_subj{i, j, k};

                    if isa(exponent_one, 'double')
                        exponent_one = num2cell(exponent_one, 1)';
                    end

                    % peak
                    peak_freq = nan(size(exponent_one));
                    peak_power = nan(size(exponent_one));
                    peak_exponent = nan(size(exponent_one));
                    for l = 1:size(exponent_one, 1)
                        temp_peak = peak_one{l};
                        temp_exponent = exponent_one{l};
                        if any(temp_peak ~= 0, 'all')
                            if length(temp_exponent) == 2 && temp_exponent(1) < temp_exponent(2)
                                [~, temp_idx] = max(temp_peak(:, 2));
                                peak_freq(l, 1) = temp_peak(temp_idx, 1);
                                peak_power(l, 1) = temp_peak(temp_idx, 2);
                                peak_exponent(l, 1) = temp_exponent(2);
                            end
                            clear temp_idx
                        end
                        clear temp_*
                    end
                    clear exponent_one peak_one

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
                    rho_sum(i, j, k, :) = rho_band;
                    pval_sum(i, j, k, :) = pval_band;
                    clear temp_* rho_band pval_band
                end
            end
        end
        rho_sum_subj = [rho_sum_subj; rho_sum];
        pval_sum_subj = [pval_sum_subj; pval_sum];
        clear rho_sum pval_sum
    else
        disp(idx_subj);
    end
end

% save data
save_path = '../data/fig1h_split_frequency_coupling.mat';
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