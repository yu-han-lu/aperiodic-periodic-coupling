%%
clear; clc; close all; warning off;
addpath('../data/tools/');

%%
% inter-ictal
data = load('../data/fig4c_inter-ictal_result.mat');
rho_inter = data.rho_sum_subj;
clear data

rho_inter_sum = [];
subj_inter_sum = [];
chan_inter_sum = [];
for i = 1:size(rho_inter, 1)
    temp_rho = rho_inter{i, 1};
    temp_subj = rho_inter{i, 3};
    temp_chan = rho_inter{i, 4};
    rho_inter_sum = [rho_inter_sum; temp_rho];
    subj_inter_sum = [subj_inter_sum; temp_subj];
    chan_inter_sum = [chan_inter_sum; temp_chan];
    clear temp_*
end

% ictal
data = load('../data/fig4c_ictal_result.mat');
rho_ictal = data.rho_sum_subj;
clear data

rho_ictal_sum = [];
subj_ictal_sum = [];
chan_ictal_sum = [];
for i = 1:size(rho_ictal, 1)
    temp_rho = rho_ictal{i, 1};
    temp_subj = rho_ictal{i, 3};
    temp_chan = rho_ictal{i, 4};
    rho_ictal_sum = [rho_ictal_sum; temp_rho];
    subj_ictal_sum = [subj_ictal_sum; temp_subj];
    chan_ictal_sum = [chan_ictal_sum; temp_chan];
    clear temp_*
end

%%
% LME: rho ~ state + (1|subject) + (1|subject:channel)
% state: interictal = reference, beta = ictal - interictal
band_sum = {'delta-theta', 'alpha', 'beta', 'low-gamma', 'high-gamma'};
formula = 'rho ~ state + (1|subject) + (1|subject:channel)';
formula0 = 'rho ~ 1 + (1|subject) + (1|subject:channel)';
results = table();
lme_sum = cell(numel(band_sum), 1);

fprintf('\nFormula: %s\n', formula);
fprintf('state: interictal = reference, beta = ictal - interictal\n\n');

for i = 1:size(rho_inter_sum, 2)
    y1 = rho_inter_sum(:, i);
    y2 = rho_ictal_sum(:, i);
    ok1 = isfinite(y1);
    ok2 = isfinite(y2);

    rho = [y1(ok1); y2(ok2)];
    state = [repmat({'interictal'}, sum(ok1), 1); repmat({'ictal'}, sum(ok2), 1)];
    subject = [subj_inter_sum(ok1); subj_ictal_sum(ok2)];
    channel = [chan_inter_sum(ok1); chan_ictal_sum(ok2)];

    tbl = table(rho, categorical(state, {'interictal', 'ictal'}), ...
        categorical(subject), categorical(channel), ...
        'VariableNames', {'rho', 'state', 'subject', 'channel'});

    lme = fitlme(tbl, formula);
    lme0 = fitlme(tbl, formula0);
    cmp = compare(lme0, lme);
    lme_sum{i} = lme;

    cn = lme.CoefficientNames(:);
    idx = find(strcmp(cn, 'state_ictal'), 1);
    b = lme.Coefficients.Estimate;
    se = lme.Coefficients.SE;
    tstat = lme.Coefficients.tStat;
    p = lme.Coefficients.pValue;
    df = lme.Coefficients.DF;
    ci = coefCI(lme);

    n_inter = sum(ok1);
    n_ictal = sum(ok2);
    n_subj = numel(categories(removecats(tbl.subject)));
    mean_inter = mean(y1(ok1));
    mean_ictal = mean(y2(ok2));

    row = table(band_sum(i), n_inter, n_ictal, n_subj, mean_inter, mean_ictal, ...
        b(idx), se(idx), tstat(idx), df(idx), p(idx), ci(idx, 1), ci(idx, 2), ...
        cmp.LRStat(2), cmp.deltaDF(2), cmp.pValue(2), ...
        'VariableNames', {'band', 'n_interictal', 'n_ictal', 'n_subject', ...
        'mean_interictal', 'mean_ictal', ...
        'beta', 'se', 't', 'df', 'p', 'ci_lo', 'ci_hi', ...
        'lrt_LR', 'lrt_df', 'lrt_p'});
    if isempty(results)
        results = row;
    else
        results = [results; row]; %#ok<AGROW>
    end

    fprintf('==== %s ====\n', band_sum{i});
    fprintf('N = %d interictal / %d ictal, %d subjects\n', n_inter, n_ictal, n_subj);
    fprintf('mean rho: interictal = %.4f, ictal = %.4f\n', mean_inter, mean_ictal);
    fprintf('state_ictal: beta = %.4f, SE = %.4f, t = %.3f, df = %.1f, p = %.4g, 95CI [%.4f, %.4f]\n', ...
        b(idx), se(idx), tstat(idx), df(idx), p(idx), ci(idx, 1), ci(idx, 2));
    fprintf('LRT vs intercept-only: LR = %.3f, df = %d, p = %.4g\n\n', ...
        cmp.LRStat(2), cmp.deltaDF(2), cmp.pValue(2));
end

[~, ~, p_fdr] = fdr_bh(results.p(:), .05, 'dep');
results.p_fdr = min(p_fdr, 1);