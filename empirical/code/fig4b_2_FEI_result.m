%%
clear; clc; close all; warning off;

%% 
% input data
load('../data/fig4a_FEI_data.mat');

% plot pre-ictal and ictal
FEI_pre_mean = nanmean(FEI_pre, 2);
FEI_pre_sem = squeeze(nanstd(FEI_pre, 0, 2) ./ sqrt(size(FEI_pre, 2)));
FEI_ictal_mean = nanmean(FEI_ictal, 2);
FEI_ictal_sem = squeeze(nanstd(FEI_ictal, 0, 2) ./ sqrt(size(FEI_ictal, 2)));
FEI_mean = [FEI_pre_mean; FEI_ictal_mean];
FEI_sem = [FEI_pre_sem; FEI_ictal_sem];

t1 = (8.75:2.5:5*60-8.75)-5*60;
t2 = 8.75:2.5:5*60-8.75;
t = [t1, t2];

x1 = t1(:)./60;
y1 = FEI_pre_mean(:);
X1 = [ones(length(y1), 1), x1];
Y1 = y1;
[b1,~,~,~,~] = regress(Y1, X1);
z1 = b1(1) + b1(2) * x1;

figure;
hold on
plot(t./60, FEI_mean, 'k');
plot_shade(t./60, FEI_mean', FEI_sem', 'k');
plot(x1, z1, 'r');
xline(0);
xlabel('time (min)');
ylabel('FEI');
title('pre-ictal and ictal');
ylim([1 1.2]);

mdl = fitlm(t1, FEI_pre_mean);
disp(mdl);

% plot inter-ictal
FEI_inter_mean = nanmean(FEI_inter, 2);
FEI_inter_sem = squeeze(nanstd(FEI_inter, 0, 2) ./ sqrt(size(FEI_inter, 2)));

t = 8.75:2.5:20*60-8.75;
figure;
hold on
plot(t./60, FEI_inter_mean, 'k');
plot_shade(t./60, FEI_inter_mean', FEI_inter_sem', 'k');
xlabel('time (min)');
ylabel('FEI');
title('inter-ictal');
ylim([1 1.2]);

% plot
figure;
hold on
dyn_draw_bar([1], squeeze(nanmean(FEI_inter, 1))');
dyn_draw_bar([2], squeeze(nanmean(FEI_pre, 1))');
dyn_draw_bar([3], squeeze(nanmean(FEI_ictal, 1))');
axis square
xlim([0 4]);
ylim([0.9 1.2]);

%%
% LME: FEI ~ state + (1|subject) + (1|subject:channel)
% one value per channel (time-averaged), matching the bar plot
addpath('/home/hangze/Desktop/tools/');

y_inter = squeeze(nanmean(FEI_inter, 1))';
y_pre = squeeze(nanmean(FEI_pre, 1))';
y_ictal = squeeze(nanmean(FEI_ictal, 1))';

state_name = {'interictal', 'preictal', 'ictal'};
y_sum = {y_inter, y_pre, y_ictal};
subj_sum = {subject_inter, subject_pre, subject_ictal};
chan_sum = {channel_inter, channel_pre, channel_ictal};

formula = 'FEI ~ state + (1|subject) + (1|subject:channel)';
formula0 = 'FEI ~ 1 + (1|subject) + (1|subject:channel)';

FEI = [];
state = {};
subject = [];
channel = [];
n_state = zeros(1, 3);
mean_state = nan(1, 3);
for i = 1:3
    y = y_sum{i};
    ok = isfinite(y);
    n_state(i) = sum(ok);
    mean_state(i) = mean(y(ok));
    FEI = [FEI; y(ok)]; %#ok<AGROW>
    state = [state; repmat(state_name(i), sum(ok), 1)]; %#ok<AGROW>
    subject = [subject; subj_sum{i}(ok)]; %#ok<AGROW>
    channel = [channel; chan_sum{i}(ok)]; %#ok<AGROW>
end

tbl = table(FEI, categorical(state, state_name), ...
    categorical(subject), categorical(channel), ...
    'VariableNames', {'FEI', 'state', 'subject', 'channel'});

lme = fitlme(tbl, formula);
lme0 = fitlme(tbl, formula0);
cmp = compare(lme0, lme);
disp(lme);

stats = anova(lme);
keep = ~strcmp(cellstr(stats.Term), '(Intercept)');
disp(stats(keep, :));

n_subj = numel(categories(removecats(tbl.subject)));
fprintf('\nFormula: %s\n', formula);
fprintf('state: interictal = reference\n');
fprintf('N = %d interictal / %d preictal / %d ictal, %d subjects\n', ...
    n_state(1), n_state(2), n_state(3), n_subj);
fprintf('mean FEI: interictal = %.4f, preictal = %.4f, ictal = %.4f\n', ...
    mean_state(1), mean_state(2), mean_state(3));
fprintf('LRT vs intercept-only: LR = %.3f, df = %d, p = %.4g\n\n', ...
    cmp.LRStat(2), cmp.deltaDF(2), cmp.pValue(2));

% pairwise two-state LME
pair_idx = [1 2; 1 3; 2 3];
results = table();
lme_sum = cell(size(pair_idx, 1), 1);
for i = 1:size(pair_idx, 1)
    a = pair_idx(i, 1);
    b = pair_idx(i, 2);
    y1 = y_sum{a};
    y2 = y_sum{b};
    ok1 = isfinite(y1);
    ok2 = isfinite(y2);

    FEI = [y1(ok1); y2(ok2)];
    state = [repmat(state_name(a), sum(ok1), 1); repmat(state_name(b), sum(ok2), 1)];
    subject = [subj_sum{a}(ok1); subj_sum{b}(ok2)];
    channel = [chan_sum{a}(ok1); chan_sum{b}(ok2)];

    tbl = table(FEI, categorical(state, state_name([a b])), ...
        categorical(subject), categorical(channel), ...
        'VariableNames', {'FEI', 'state', 'subject', 'channel'});

    lme_pair = fitlme(tbl, formula);
    lme0_pair = fitlme(tbl, formula0);
    cmp_pair = compare(lme0_pair, lme_pair);
    lme_sum{i} = lme_pair;

    cn = lme_pair.CoefficientNames(:);
    idx = find(strcmp(cn, sprintf('state_%s', state_name{b})), 1);
    beta = lme_pair.Coefficients.Estimate;
    se = lme_pair.Coefficients.SE;
    tstat = lme_pair.Coefficients.tStat;
    p = lme_pair.Coefficients.pValue;
    df = lme_pair.Coefficients.DF;
    ci = coefCI(lme_pair);

    n1 = sum(ok1);
    n2 = sum(ok2);
    n_subj_pair = numel(categories(removecats(tbl.subject)));
    mean1 = mean(y1(ok1));
    mean2 = mean(y2(ok2));
    contrast = sprintf('%s - %s', state_name{b}, state_name{a});

    row = table({contrast}, n1, n2, n_subj_pair, mean1, mean2, ...
        beta(idx), se(idx), tstat(idx), df(idx), p(idx), ci(idx, 1), ci(idx, 2), ...
        cmp_pair.LRStat(2), cmp_pair.deltaDF(2), cmp_pair.pValue(2), ...
        'VariableNames', {'contrast', 'n_ref', 'n_cmp', 'n_subject', ...
        'mean_ref', 'mean_cmp', ...
        'beta', 'se', 't', 'df', 'p', 'ci_lo', 'ci_hi', ...
        'lrt_LR', 'lrt_df', 'lrt_p'});
    if isempty(results)
        results = row;
    else
        results = [results; row]; %#ok<AGROW>
    end

    fprintf('==== %s vs %s ====\n', state_name{b}, state_name{a});
    fprintf('N = %d %s / %d %s, %d subjects\n', n1, state_name{a}, n2, state_name{b}, n_subj_pair);
    fprintf('mean FEI: %s = %.4f, %s = %.4f\n', state_name{a}, mean1, state_name{b}, mean2);
    fprintf('state_%s: beta = %.4f, SE = %.4f, t = %.3f, df = %.1f, p = %.4g, 95CI [%.4f, %.4f]\n', ...
        state_name{b}, beta(idx), se(idx), tstat(idx), df(idx), p(idx), ci(idx, 1), ci(idx, 2));
    fprintf('LRT vs intercept-only: LR = %.3f, df = %d, p = %.4g\n\n', ...
        cmp_pair.LRStat(2), cmp_pair.deltaDF(2), cmp_pair.pValue(2));
end

[~, ~, p_fdr] = fdr_bh(results.p(:), .05, 'dep');
results.p_fdr = min(p_fdr, 1);