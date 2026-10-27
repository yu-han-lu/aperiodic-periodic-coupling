%% Generate lme_tvalue_heatmap (5 bands x 25 conditions; MATLAB visualization)
% Revised: natpal('prgn', 256); clim([-25 25])
clear; clc; close all; warning off;

%% ---- input .mat (change these two paths) ----
rho_mat = '../data/fig1e_split_frequency_coupling.mat';
subject_channel_mat = '../data/fig1e_split_subj.mat';

%% ---------------------------------------------
% subject_channel_sort.mat must contain:
%   subject_sum  : nObs x 1 subject IDs
%   channel_sum  : nObs x 1 channel IDs
% nObs must equal size(rho_sum_subj, 1)

rho_struct = load(rho_mat);
sc_struct  = load(subject_channel_mat);
rho_sum_subj = rho_struct.rho_sum_subj;
subj = sc_struct.subject_sum;
chan = sc_struct.channel_sum;

nObs = size(rho_sum_subj, 1);
if numel(subj) ~= nObs || numel(chan) ~= nObs
    error('subject/channel length (%d/%d) must match rho first dim (%d).', ...
        numel(subj), numel(chan), nObs);
end

window = [1 2 5 10 30];
average = [2 4 6 8 10];
band_names = {'Theta', 'Alpha', 'Beta', 'Low-gamma', 'High-gamma'};

t_values = nan(5, 5, 5);
for j = 1:5
    for k = 1:5
        for i = 1:5
            % rho_sum_subj: channel x window x average x band
            vals = squeeze(rho_sum_subj(:, j, k, i));
            ok = ~isnan(vals) & ~isnan(subj) & ~isnan(chan);
            if ~any(ok)
                continue;
            end
            tbl = table(vals(ok), categorical(subj(ok)), categorical(chan(ok)), ...
                'VariableNames', {'rho', 'subject', 'channel'});
            try
                lme = fitlme(tbl, 'rho ~ 1 + (1|subject) + (1|subject:channel)');
                t_values(i, j, k) = lme.Coefficients.tStat(1);
            catch
            end
        end
    end
end

col_labels = cell(1, 25);
col_idx = 0;
for j = 1:5
    for k = 1:5
        col_idx = col_idx + 1;
        col_labels{col_idx} = sprintf('w=%d,a=%d', window(j), average(k));
    end
end

t_mat = nan(5, 25);
for j = 1:5
    for k = 1:5
        col = (j - 1) * 5 + k;
        t_mat(:, col) = t_values(:, j, k);
    end
end

fig = make_lme_tvalue_heatmap(t_mat, band_names, col_labels, true);

function fig = make_lme_tvalue_heatmap(t_mat, band_names, col_labels, visible)
    if visible
        fig = figure('Color', 'w', 'Units', 'pixels', 'Position', [100 100 820 420]);
    else
        fig = figure('Color', 'w', 'Units', 'pixels', 'Position', [100 100 820 420], 'Visible', 'off');
    end

    cmap = natpal('prgn', 256);
    t_lim = 25;

    ax_hm = axes('Parent', fig);
    hold(ax_hm, 'on');
    draw_heatmap_cells(ax_hm, t_mat, t_lim, cmap);
    colormap(ax_hm, cmap);
    clim(ax_hm, [-25 25]);

    set(ax_hm, 'YDir', 'reverse', ...
        'XTick', 1:25, 'XTickLabel', col_labels, 'XTickLabelRotation', 45, ...
        'YTick', 1:5, 'YTickLabel', band_names, ...
        'TickLabelInterpreter', 'none', 'FontSize', 9);
    xlabel(ax_hm, 'Window / Average');
    ylabel(ax_hm, 'Band');
    title(ax_hm, 'LME intercept t-value (rho \neq 0)');

    x_span = 25;
    y_span = 5;
    hm_wh_ratio = 2;
    pbaspect(ax_hm, [x_span / (hm_wh_ratio * y_span), 1, 1]);
    axis(ax_hm, 'manual');

    ax_hm.ActivePositionProperty = 'position';
    hm_w = 480;
    hm_h = hm_w / hm_wh_ratio;
    ax_hm.Units = 'pixels';
    ax_hm.Position = [70 95 hm_w hm_h];

    ax_cb = axes('Parent', fig, 'Units', 'pixels', 'Position', ...
        [70 + hm_w + 12, 95, 28, hm_h]);
    draw_colorbar(ax_cb, t_lim, cmap);
    colormap(ax_cb, cmap);
    clim(ax_cb, [-25 25]);
end


function draw_heatmap_cells(ax, t_mat, t_lim, cmap)
    [n_rows, n_cols] = size(t_mat);
    for row = 1:n_rows
        for col = 1:n_cols
            val = t_mat(row, col);
            if isnan(val)
                continue;
            end
            rectangle(ax, 'Position', [col - 0.5, row - 0.5, 1, 1], ...
                'FaceColor', tvalue_to_rgb(val, t_lim, cmap), 'EdgeColor', 'none');
        end
    end
    xlim(ax, [0.5, n_cols + 0.5]);
    ylim(ax, [0.5, n_rows + 0.5]);
end


function draw_colorbar(ax, t_lim, cmap)
    n_seg = size(cmap, 1);
    hold(ax, 'on');
    for i = 1:n_seg
        val = -t_lim + ((i - 0.5) / n_seg) * (2 * t_lim);
        rectangle(ax, 'Position', [0, i - 1, 1, 1], ...
            'FaceColor', tvalue_to_rgb(val, t_lim, cmap), 'EdgeColor', 'none');
    end
    xlim(ax, [0, 1]);
    ylim(ax, [0, n_seg]);
    set(ax, 'XTick', [], 'YTick', linspace(0, n_seg, 5), ...
        'YTickLabel', compose('%.0f', linspace(-t_lim, t_lim, 5)), ...
        'TickDir', 'out', 'Box', 'off', 'FontSize', 9);
    ylabel(ax, 't-value');
end


function rgb = tvalue_to_rgb(val, t_lim, cmap)
    n = size(cmap, 1);
    frac = (val + t_lim) / (2 * t_lim);
    frac = max(0, min(1, frac));
    idx = round(frac * (n - 1)) + 1;
    rgb = cmap(idx, :);
end

function cmap = natpal(name, n)
%NATPAL ColorBrewer palette wrapper (via brewermap).
%   cmap = natpal('prgn', 256)
    if nargin < 2 || isempty(n)
        n = 256;
    end
    brewermap_dir = '/home/hangze/Downloads/fieldtrip-20250106/external/brewermap';
    if ~exist('brewermap', 'file')
        addpath(brewermap_dir);
    end
    cmap = brewermap(n, char(name));
end
