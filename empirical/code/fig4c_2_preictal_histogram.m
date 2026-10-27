%%
clear; clc; close all; warning off;

%%
% inter-ictal
data = load('../data/fig4c_inter-ictal_result.mat');
rho_inter = data.rho_sum_subj;
clear data

rho_inter_sum = [];
pval_inter_sum = [];
for i = 1:size(rho_inter, 1)
    temp_rho = rho_inter{i, 1};
    temp_pval = rho_inter{i, 2};
    rho_inter_sum = [rho_inter_sum; temp_rho];
    pval_inter_sum = [pval_inter_sum; temp_pval];
    clear temp_*
end

% pre-ictal
data = load('../data/fig4c_pre-ictal_result.mat');
rho_preictal = data.rho_sum_subj;
clear data

rho_preictal_sum = [];
pval_preictal_sum = [];
for i = 1:size(rho_preictal, 1)
    temp_rho = rho_preictal{i, 1};
    temp_pval = rho_preictal{i, 2};
    rho_preictal_sum = [rho_preictal_sum; temp_rho];
    pval_preictal_sum = [pval_preictal_sum; temp_pval];
    clear temp_*
end

ylim_idx = [60 60 90 40 40];
figure;
t = 0;
for i = 1:size(rho_inter_sum, 2)
    t = t + 1;
    subplot(4, 5, t);
    hold on
    histogram(rho_inter_sum(:, i), 20, 'FaceColor', 'k');
    xline(nanmean(rho_inter_sum(:, i)), 'k');
    axis square
    xlim([-1 1]);
    ylim([0 ylim_idx(i)]);
end

for i = 1:size(rho_preictal_sum, 2)
    t = t + 1;
    subplot(4, 5, t);
    hold on
    histogram(rho_preictal_sum(:, i), 20, 'FaceColor', 'r');
    xline(nanmean(rho_preictal_sum(:, i)), 'r');
    axis square
    xlim([-1 1]);
    ylim([0 ylim_idx(i)]);
end
