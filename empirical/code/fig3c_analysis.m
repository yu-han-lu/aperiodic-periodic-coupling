%%
clear; clc; close all; warning off;

%%
data = load('../data/fig3c_oscillation_data.mat');
peak_whole = data.peak;
clear data

% IRASA leftover stim artifact: if during-stim has a peak in these bands,
% nan that electrode for all states of this condition
% row1 = C2/C3 ~20 Hz line; row2 = C1 ~16 Hz residual (comment out to keep C1)
exclude_bands = [19.5 21.0;
                 15.5 18.0];

% peak bands: [lo hi]; last edge inclusive, others [lo, hi)
freq_bands = [1 8; 8 13; 13 30];
band_name = {'delta-theta', 'alpha', 'beta'};
y_lims = [8 15 30];
nBand = size(freq_bands, 1);

% loop-subject
peak_freq_subj = [];
subject_sum = [];
t = 0;
for sub = 1:length(peak_whole)
    peak_sum = squeeze(peak_whole{sub});

    % main
    peak_freq = nan([size(peak_sum) nBand]);
    for i = 1:size(peak_sum, 1)
        for j = 1:size(peak_sum, 2)
            for k  = 1:size(peak_sum, 3)
                temp_peak = peak_sum{i, j, k};
                if ~all(temp_peak == 0)
                    for b = 1:nBand
                        if b < nBand
                            in_band = temp_peak(:, 1) >= freq_bands(b, 1) & temp_peak(:, 1) < freq_bands(b, 2);
                        else
                            in_band = temp_peak(:, 1) >= freq_bands(b, 1) & temp_peak(:, 1) <= freq_bands(b, 2);
                        end
                        if any(in_band)
                            temp_freq = temp_peak(in_band, 1);
                            temp_pow = temp_peak(in_band, 2);
                            [~, temp_idx] = max(temp_pow);
                            peak_freq(i, j, k, b) = temp_freq(temp_idx);
                        end
                        clear in_band temp_freq temp_pow temp_idx
                    end
                end
                clear temp_*
            end
        end
    end

    % nan electrodes that show stim-artifact peaks during stim (state 2)
    for i = 1:size(peak_sum, 1)
        for k = 1:size(peak_sum, 3)
            temp_peak = peak_sum{i, 2, k};
            if ~all(temp_peak == 0)
                is_artifact = false;
                for b = 1:size(exclude_bands, 1)
                    if any(temp_peak(:, 1) >= exclude_bands(b, 1) & ...
                            temp_peak(:, 1) <= exclude_bands(b, 2))
                        is_artifact = true;
                        break
                    end
                end
                if is_artifact
                    peak_freq(i, :, k, :) = nan;
                end
                clear is_artifact
            end
            clear temp_*
        end
    end
    
    % remove post-stim
    peak_freq = peak_freq(:, 1:2, :, :);

    % add a new condition (pre and during both ~isnan), per band
    for i = 1:size(peak_freq, 1)
        for j = 1:size(peak_freq, 3)
            for b = 1:nBand
                temp_peak_freq = squeeze(peak_freq(i, :, j, b));
                if (isnan(temp_peak_freq(1)) && ~isnan(temp_peak_freq(2))) || (isnan(temp_peak_freq(2)) && ~isnan(temp_peak_freq(1)))
                    temp_peak_freq(1:2) = nan;
                    peak_freq(i, :, j, b) = temp_peak_freq;
                end
                clear temp_*
            end
        end
    end

    % subject
    peak_freq_mean = squeeze(nanmean(peak_freq, 3));
    clear peak_freq

    t = t + 1;
    peak_freq_subj(t, :, :, :) = peak_freq_mean;
    clear peak_freq_mean
    
    subject_sum = [subject_sum; sub];
end

% plot
figure;
for b = 1:nBand
    subplot(nBand, 2, (b - 1) * 2 + 1);
    hold on
    dyn_draw_bar([1], squeeze(peak_freq_subj(:, 1, 1, b)));
    dyn_draw_bar([2], squeeze(peak_freq_subj(:, 2, 1, b)));
    dyn_draw_bar([3], squeeze(peak_freq_subj(:, 3, 1, b)));
    xlim([0 4]);
    ylim([0 y_lims(b)]);
    axis square
    title(['pre-stim  ' band_name{b}]);
    xticks(1:3);
    xticklabels({'white', 'brown', 'pink'});
    ylabel('peak frequency');

    subplot(nBand, 2, (b - 1) * 2 + 2);
    hold on
    dyn_draw_bar([1], squeeze(peak_freq_subj(:, 1, 2, b)));
    dyn_draw_bar([2], squeeze(peak_freq_subj(:, 2, 2, b)));
    dyn_draw_bar([3], squeeze(peak_freq_subj(:, 3, 2, b)));
    xlim([0 4]);
    ylim([0 y_lims(b)]);
    axis square
    title(['during-stim  ' band_name{b}]);
    xticks(1:3);
    xticklabels({'white', 'brown', 'pink'});
    ylabel('peak frequency');
end

% lme model
formula = 'peak_freq ~ state * condition * band + (1|subject)';
[nSubj, nCond, nState, ~] = size(peak_freq_subj);

peak_freq = zeros(nSubj * nCond * nState * nBand, 1);
state = zeros(nSubj * nCond * nState * nBand, 1);
condition = zeros(nSubj * nCond * nState * nBand, 1);
band = zeros(nSubj * nCond * nState * nBand, 1);
subject = zeros(nSubj * nCond * nState * nBand, 1);

idx = 1;
for s = 1:nSubj
    for c = 1:nCond
        for st = 1:nState
            for b = 1:nBand
                peak_freq(idx) = peak_freq_subj(s, c, st, b);
                condition(idx) = c;
                state(idx) = st;
                band(idx) = b;
                subject(idx) = subject_sum(s);
                idx = idx + 1;
            end
        end
    end
end

ok = isfinite(peak_freq);
tbl = table(peak_freq(ok), categorical(state(ok)), categorical(condition(ok)), ...
    categorical(band(ok)), categorical(subject(ok)), ...
    'VariableNames', {'peak_freq', 'state', 'condition', 'band', 'subject'});

lme = fitlme(tbl, formula);
disp(lme);

stats = anova(lme);
keep = ~strcmp(cellstr(stats.Term), '(Intercept)');
disp(stats(keep, :));
