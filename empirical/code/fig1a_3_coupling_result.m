%%
clear; clc; close all; warning off;

%%
sublist = [22 28 65 68 69 70 71 72 73 74 75 76 77 78 85 89 92 96 97 98 117 123 131 135 140 141 144 152 153 159 164 ...
    168 169 176 177 186 189 190 192 197 198 201 204 205 213 216 229 241 245 246 253 255 256 257 258 259 260 261 262 263 264];

% loop-subject
rho_sum_subj = [];
pval_sum_subj = [];
for idx_subj = sublist
    % loop-window
    rho_sum = [];
    pval_sum = [];
    for i = 1:5
        % loop-average
        for j = 1:5
            input_path = sprintf('../data/fooof_data/sub-%d/sub-%d-%d-%d.mat', idx_subj, idx_subj, i, j);
            data = load(input_path);
            peak_sum = data.peak_params_sum;
            exponent_sum = cell2mat(data.fit_exp_sum);
            clear data

            % peak
            peak_freq = nan(size(peak_sum));
            peak_power = nan(size(peak_sum));
            peak_exponent = nan(size(peak_sum));
            for k = 1:size(peak_sum, 1)
                for l = 1:size(peak_sum, 2)
                    temp_peak = peak_sum{k, l};
                    temp_exponent = exponent_sum(k, l);
                    if any(temp_peak ~= 0, 'all') && ~isnan(temp_exponent)
                        [~, temp_idx] = max(temp_peak(:, 2)); % need to revise
                        peak_freq(k, l) = temp_peak(temp_idx, 1);
                        peak_power(k, l) = temp_peak(temp_idx, 2);
                        peak_exponent(k, l) = temp_exponent;
                        clear temp_idx
                    end
                    clear temp_*
                end
            end

            % correlation
            rho_channel = nan(size(peak_exponent, 1), 5);
            pval_channel = nan(size(peak_exponent, 1), 5);
            for k = 1:size(peak_exponent, 1)
                temp_exponent = squeeze(peak_exponent(k, :))';
                temp_frequency = squeeze(peak_freq(k, :))';
                temp_exponent = temp_exponent(~isnan(temp_exponent));
                temp_frequency = temp_frequency(~isnan(temp_frequency));

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
                        rho_channel(k, l) = rho;
                        pval_channel(k, l) = pval;
                        clear rho pval
                    end
                    clear temp_idx temp_exponent_* temp_frequency_*
                end
                clear temp_*
            end
            rho_sum(:, :, i, j) = rho_channel;
            pval_sum(:, :, i, j) = pval_channel;
            clear rho_channel pval_channel
        end
    end
    rho_sum_subj = [rho_sum_subj; rho_sum];
    pval_sum_subj = [pval_sum_subj; pval_sum];
    clear rho_sum pval_sum
end

% save data
save_path = '../data/fig1a_fooof_frequency_coupling.mat';
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