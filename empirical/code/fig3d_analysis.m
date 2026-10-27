%%
clear; clc; close all; warning off;

%% 
data = load('../data/fig3d_split_data.mat');
exponent_whole = data.exponent;
knee_whole = data.knee;
clear data

exponent_subj = [];
subject_sum = [];
t = 0;
for sub = 1:length(exponent_whole)
    exponent = exponent_whole{sub, 1};
    knee = knee_whole{sub, 1};

    % loop
    exponent_sum = nan(length(exponent{1, 1}), size(exponent, 1), size(exponent, 2));
    for i = 1:size(exponent, 1)
        for j = 1:size(exponent, 2)
            temp_exponent = exponent{i, j};
            temp_knee = knee{i, j};
            for k = 1:length(temp_exponent)
                temp_exponent_one = temp_exponent{k};
                temp_knee_one = temp_knee{k};
                if length(temp_exponent_one) == 2 && temp_exponent_one(1) < temp_exponent_one(2) && temp_knee_one < 10
                    exponent_sum(k, i, j) =  temp_exponent_one(2);
                end
            end
            clear temp_*
        end
    end

    % add a new condition (pre and during both ~isnan)
    for i = 1:size(exponent_sum, 2)
        for j = 1:size(exponent_sum, 1)
            temp_exponent = squeeze(exponent_sum(j, i, :));
            if (isnan(temp_exponent(1)) && ~isnan(temp_exponent(2))) || (isnan(temp_exponent(2)) && ~isnan(temp_exponent(1)))
                temp_exponent(1:2) = nan;
                exponent_sum(j, i, :) = temp_exponent;
            end
            clear temp_*
        end
    end

    % subject
    exponent_mean = squeeze(nanmean(exponent_sum, 1));
    clear exponent exponent_sum

    t = t + 1;
    exponent_subj(t, :, :) = exponent_mean;
    clear exponent_mean

    subject_sum = [subject_sum; sub];
end

% plot
figure;
subplot(1, 2, 1);
hold on
dyn_draw_bar([1], exponent_subj(:, 1, 1));
dyn_draw_bar([2], exponent_subj(:, 2, 1));
dyn_draw_bar([3], exponent_subj(:, 3, 1));
dyn_draw_bar([4], exponent_subj(:, 4, 1));
axis square
xlim([0 5]);
ylim([1 3]);
title('pre-stim');
xticks(1:4);
xticklabels({'8 Hz', '32 Hz', '64 Hz', '128 Hz'});
ylabel('exponent');

subplot(1, 2, 2);
hold on
dyn_draw_bar([1], exponent_subj(:, 1, 2));
dyn_draw_bar([2], exponent_subj(:, 2, 2));
dyn_draw_bar([3], exponent_subj(:, 3, 2));
dyn_draw_bar([4], exponent_subj(:, 4, 2));
axis square
xlim([0 5]);
ylim([1 3]);
title('during-stim');
xticks(1:4);
xticklabels({'8 Hz', '32 Hz', '64 Hz', '128 Hz'});
ylabel('exponent');

% bootstrap
t = 1;
Bv = 1e4;
for j = 1:size(exponent_subj, 2)-1
    for k = j+1:size(exponent_subj, 2)
        group1 = exponent_subj(:, j, 2);
        group2 = exponent_subj(:, k, 2);

        max_len = max(length(group1), length(group2));
        group1_padded = nan(max_len, 1);
        group2_padded = nan(max_len, 1);
        group1_padded(1:length(group1)) = group1;
        group2_padded(1:length(group2)) = group2;

        dx = [group1_padded, group2_padded];
        [O, A] = bootstrap_for_vector(Bv, 0.05, @compare_mean, dx);
        px(t) = 2 * (min(sum(A <= 0), sum(A >= 0)) + 1) / (Bv + 1);

        t = t + 1;
        clear group1* group2* dx
    end
end

[h, crit_p, p_fdr] = fdr_bh(px);


% lme model
formula = 'exponent ~ state * condition + (1|subject)';
[nSubj, nCond, nState] = size(exponent_subj);

exponent = zeros(nSubj * nCond * nState, 1);
state = zeros(nSubj * nCond * nState, 1);
condition = zeros(nSubj * nCond * nState, 1);
subject = zeros(nSubj * nCond * nState, 1);

idx = 1;
for s = 1:nSubj
    for c = 1:nCond
        for st = 1:nState
            exponent(idx) = exponent_subj(s, c, st);
            condition(idx) = c;
            state(idx) = st;
            subject(idx) = subject_sum(s);
            idx = idx + 1;
        end
    end
end

ok = isfinite(exponent);
tbl = table(exponent(ok), categorical(state(ok)), categorical(condition(ok)), categorical(subject(ok)), ...
    'VariableNames', {'exponent', 'state', 'condition', 'subject'});

lme = fitlme(tbl, formula);
disp(lme);

stats = anova(lme);
keep = ~strcmp(cellstr(stats.Term), '(Intercept)');
disp(stats(keep, :));
