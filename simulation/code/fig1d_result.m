%%
clear; clc; close all; warning off;
addpath('../../tools/');

%% 
% plot 1
load('../data/fig1c_irasa_data.mat');

figure;
subplot(1, 3, 1);
plot(log10(f), log10(original_powspectra));
axis square
xlim([log10(0.2) log10(200)]);

% plot 2
data = load('../data/fig1d_fooof_result.mat', 'peak_cf', 'aperiodic_exponent');
exponent_fooof = data.aperiodic_exponent';
oscillation_fooof = data.peak_cf';

[rho, pval] = corr(oscillation_fooof, exponent_fooof, 'Type', 'Spearman', 'Rows', 'pairwise');

subplot(1, 3, 2);
hz_draw_scatter(oscillation_fooof, exponent_fooof);
ylim([1.9 1.98]);
xlabel('frequency');
ylabel('fooof exponent');
xlim([30 150]);
title(['r (' num2str(rho) '); p (' num2str(pval) ')']);
axis square

% plot 3
load('../data/fig1d_split_exponent_result.mat');
load('../data/fig1d_split_oscillation_result.mat');

exponent_split = cell2mat(exponent_sum);
oscillation_split = nan(100, 3);
for i = 1:length(peak_sum)
    if any(cell2mat(peak_sum(i)) ~= 0, 'all')
        oscillation_split(i, :) = cell2mat(peak_sum(i));
    end
end
clear exponent_sum knee_freq_sum peak_sum

[rho, pval] = corr(oscillation_split(:, 1), exponent_split, 'Type', 'Spearman', 'Rows', 'pairwise');

subplot(1, 3, 3);
hz_draw_scatter(oscillation_split(:, 1), exponent_split);
xlabel('frequency');
ylabel('split exponent');
xlim([30 150]);
ylim([1.86 2.02]);
title(['r (' num2str(rho) '); p (' num2str(pval) ')']);
axis square