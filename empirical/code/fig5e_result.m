%%
clear; clc; close all; warning off;
addpath('../../tools/');

%%
% input data
data = load('../data/fig5_CCEP+_data.mat');
rho_direct_pre = data.rho_sum_pre;
rho_direct_post = data.rho_sum_post;
subject_direct = data.subject_sum(:);
channel_direct = data.channel_sum(:);
clear data

data = load('../data/fig5_CCEP-_data.mat');
rho_none_direct_pre = data.rho_sum_pre;
rho_none_direct_post = data.rho_sum_post;
subject_none_direct = data.subject_sum(:);
channel_none_direct = data.channel_sum(:);
clear data

% delta
rho_direct_delta = abs(rho_direct_pre - rho_direct_post);
rho_none_direct_delta = abs(rho_none_direct_pre - rho_none_direct_post);

% plot
band_sum = {'delta-theta', 'alpha', 'beta', 'low-gamma', 'high-gamma'};
co = colororder;
figure;
for i = 1:size(rho_direct_delta, 2)
    subplot(1, 5, i);
    hold on

    y1 = rho_direct_delta(:, i);
    y2 = rho_none_direct_delta(:, i);
    y1 = y1(isfinite(y1));
    y2 = y2(isfinite(y2));

    dyn_draw_bar([1], y1);
    dyn_draw_bar([2], y2);
    set(findobj(gca, 'Type', 'Bar'), 'FaceAlpha', 0.35);

    swarmchart(ones(size(y1)), y1, 14, co(1, :), 'filled', ...
        'XJitterWidth', 0.35, 'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
    swarmchart(2 * ones(size(y2)), y2, 14, co(2, :), 'filled', ...
        'XJitterWidth', 0.35, 'MarkerFaceAlpha', 0.4, 'MarkerEdgeColor', 'none');
    uistack(findobj(gca, 'Type', 'ErrorBar'), 'top');

    axis square
    xlim([0 3]);
    title(band_sum{i});
    xticks(1:2);
    xticklabels({'direct', 'none-direct'});
    ylim([0 1.5]);
end

%%
% LME: delta ~ group + (1|subject) + (1|subject:channel)
formula = 'delta ~ group + (1|subject) + (1|subject:channel)';
results = table();
lme_sum = cell(size(rho_direct_delta, 2), 1);

fprintf('\nFormula: %s\n', formula);
fprintf('group: none-direct = reference, beta = direct - none-direct\n\n');

for i = 1:size(rho_direct_delta, 2)
    y1 = rho_direct_delta(:, i);
    y2 = rho_none_direct_delta(:, i);
    ok1 = isfinite(y1);
    ok2 = isfinite(y2);

    delta = [y1(ok1); y2(ok2)];
    group = [repmat({'direct'}, sum(ok1), 1); repmat({'none-direct'}, sum(ok2), 1)];
    subject = [subject_direct(ok1); subject_none_direct(ok2)];
    channel = [strcat('direct_', string(channel_direct(ok1))); ...
        strcat('none_', string(channel_none_direct(ok2)))];

    tbl = table(delta, categorical(group, {'none-direct', 'direct'}), ...
        categorical(subject), categorical(channel), ...
        'VariableNames', {'delta', 'group', 'subject', 'channel'});

    lme = fitlme(tbl, formula);
    lme_sum{i} = lme;

    cn = lme.CoefficientNames(:);
    idx = find(strcmp(cn, 'group_direct'), 1);
    b = lme.Coefficients.Estimate;
    se = lme.Coefficients.SE;
    tstat = lme.Coefficients.tStat;
    p = lme.Coefficients.pValue;
    df = lme.Coefficients.DF;
    ci = coefCI(lme);

    n_direct = sum(ok1);
    n_none = sum(ok2);
    n_subj = numel(categories(removecats(tbl.subject)));
    mean_direct = mean(y1(ok1));
    mean_none = mean(y2(ok2));

    row = table(band_sum(i), n_direct, n_none, n_subj, mean_direct, mean_none, ...
        b(idx), se(idx), tstat(idx), df(idx), p(idx), ci(idx, 1), ci(idx, 2), ...
        'VariableNames', {'band', 'n_direct', 'n_none', 'n_subject', ...
        'mean_direct', 'mean_none', 'beta', 'se', 't', 'df', 'p', 'ci_lo', 'ci_hi'});
    if isempty(results)
        results = row;
    else
        results = [results; row]; %#ok<AGROW>
    end

    fprintf('==== %s ====\n', band_sum{i});
    fprintf('N = %d direct / %d none-direct, %d subjects\n', n_direct, n_none, n_subj);
    fprintf('mean delta: direct = %.4f, none-direct = %.4f\n', mean_direct, mean_none);
    fprintf('group_direct: beta = %.4f, SE = %.4f, t = %.3f, df = %.1f, p = %.4g, 95CI [%.4f, %.4f]\n\n', ...
        b(idx), se(idx), tstat(idx), df(idx), p(idx), ci(idx, 1), ci(idx, 2));
end

disp(results);
[~, ~, p_fdr] = fdr_bh(results.p(:), .05, 'dep');