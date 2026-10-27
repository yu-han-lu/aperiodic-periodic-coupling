%%
clear; clc; close all; warning off;

%% 
% input data
load('../data/fig5_CCEP+_data.mat');

ylim_idx = [15 15 25 8 10];

figure;
t = 0;
for i = 1:size(rho_sum_pre, 2)
    t = t + 1;
    subplot(4, 5, t);
    hold on
    temp_idx = find(~isnan(rho_sum_pre(:, i)) & ~isnan(rho_sum_post(:, i)));
    histogram(rho_sum_pre(temp_idx, i), 'FaceColor', 'k', 'BinWidth', 0.1);
    xline(nanmean(rho_sum_pre(temp_idx, i)), 'k');
    axis square
    xlim([-1 1]);
    ylim([0 ylim_idx(i)]);
    clear temp_*
end

for i = 1:size(rho_sum_pre, 2)
    t = t + 1;
    subplot(4, 5, t);
    hold on
    temp_idx = find(~isnan(rho_sum_pre(:, i)) & ~isnan(rho_sum_post(:, i)));
    histogram(rho_sum_post(temp_idx, i), 'FaceColor', 'r', 'BinWidth', 0.1);
    xline(nanmean(rho_sum_post(temp_idx, i)), 'r');
    axis square
    xlim([-1 1]);
    ylim([0 ylim_idx(i)]);
    clear temp_*
end

%%
clear; clc; warning off;

%% 
% input data
load('../data/fig5_CCEP-_data.mat');

ylim_idx = [15 15 30 10 20];

figure;
t = 0;
for i = 1:size(rho_sum_pre, 2)
    t = t + 1;
    subplot(4, 5, t);
    hold on
    temp_idx = find(~isnan(rho_sum_pre(:, i)) & ~isnan(rho_sum_post(:, i)));
    histogram(rho_sum_pre(temp_idx, i), 'FaceColor', 'k', 'BinWidth', 0.1);
    xline(nanmean(rho_sum_pre(temp_idx, i)), 'k');
    axis square
    xlim([-1 1]);
    ylim([0 ylim_idx(i)]);
    clear temp_*
end

for i = 1:size(rho_sum_pre, 2)
    t = t + 1;
    subplot(4, 5, t);
    hold on
    temp_idx = find(~isnan(rho_sum_pre(:, i)) & ~isnan(rho_sum_post(:, i)));
    histogram(rho_sum_post(temp_idx, i), 'FaceColor', 'r', 'BinWidth', 0.1);
    xline(nanmean(rho_sum_post(temp_idx, i)), 'r');
    axis square
    xlim([-1 1]);
    ylim([0 ylim_idx(i)]);
    clear temp_*
end