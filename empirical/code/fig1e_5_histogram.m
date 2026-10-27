%%
clear; clc; close all; warning off;

%%
data = load('../data/fig1e_split_frequency_coupling.mat');
rho_sum_subj = squeeze(data.rho_sum_subj(:, 3, 3, :));
pval_sum_subj = squeeze(data.pval_sum_subj(:, 3, 3, :));
clear data

% fdr
for i = 1:5
    [h, crit_p, p_fdr(:, i)] = fdr_bh(pval_sum_subj(:, i), .05, 'dep');
end

% plot
t = 0;
figure;
for i = 1:5
    t = t + 1;
    subplot(2, 5, t);
    histogram(rho_sum_subj(:, i), 20);
    xline(median(squeeze(rho_sum_subj(:, i)), 'omitnan'), 'r');
    xline(0, '--r')
    axis square
    xlim([-1 1]);
end

for i = 1:5
    t = t + 1;
    subplot(2, 5, t);
    scatter(rho_sum_subj(:, i), -log10(p_fdr(:, i)));

    p_sign = (length(find(p_fdr(:, i) < 0.05)))./length(p_fdr(~isnan(p_fdr(:, i)), i));
    title(['sig = ' num2str(p_sign*100,2) '%']);

    axis square
    yline(-log10(0.05));
    xline(0, '--k');
    xlim([-1 1]);
    ylim([-2 20]);
end