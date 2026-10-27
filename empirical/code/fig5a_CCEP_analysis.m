%%
clear; clc; close all; warning off;
addpath('../../tools/');

sublist = [1 3 4 5 6 8 9 10 11 12 13 14 15 17 20];
input_path = '../data/preproc_data/';

para_fs = 1000;
para_epoch_leng = -0.5 * para_fs + 1: 0.8 * para_fs;
para_baseline = 0.30 * para_fs + 1 : 0.48 * para_fs;
para_draw_stim = 0;

%%
for idx_subj = sublist
    % check how many sessions in each subject folder
    pathSession = dir(fullfile([input_path 'sub-' num2str(idx_subj) '/'], '*.mat'));

    % entering session process
    for idx_session = 1:length(pathSession)
        data = load([pathSession(idx_session).folder '/' pathSession(idx_session).name]);
        data_sum = data.data_sum;
        trig_name = data.trig_name;
        label = data.label;
        fs = data.fs;
        clear data

        % construct trigger
        for idx_trial = 1:length(trig_name)
            tem_stim_name = trig_name(idx_trial);
            tem_stim_ch = label == string(tem_stim_name);
            if ~any(tem_stim_ch)
                clear tem*;
                continue
            end
            tem_stim_ch_idx = find(tem_stim_ch);

            % draw the stim electrode to check
            disp(['Stimulator: ' tem_stim_name{1} ' Channel index: ' num2str(tem_stim_ch_idx)]);

            % set a threshold
            tem_onsets = [];
            para_threshold = 20000;
            while length(tem_onsets) < 40 || length(tem_onsets) > 48
                tem_sig = data_sum(:, tem_stim_ch_idx, idx_trial);
                tem_sig_diff = diff(tem_sig);
                tem_above_idx = find(tem_sig_diff(11*fs+1:60*fs) > para_threshold | tem_sig_diff(11*fs+1:60*fs) < -para_threshold);
                tem_above_idx = tem_above_idx+11*fs;
                try
                    tem_onsets = tem_above_idx([true; diff(tem_above_idx) > 1]);
                end
                para_threshold = para_threshold - 100;

                if para_threshold == 0
                    disp('No oncset detected, jumping to the next...');
                    break;
                end
            end
            if para_threshold <= 0
                continue; end

            if para_draw_stim
                figure;
                subplot(3, 1, 1);
                hold on
                plot(tem_sig);
                plot(tem_above_idx, 2000*ones(1,length(tem_above_idx)), 'go');
                title([string(label(tem_stim_ch_idx))]);
            end

            % EXTRACT CCEP FROM HERE
            for idx_epoch = 1:length(tem_onsets)
                tem_epoch(:,:,idx_epoch) = data_sum(...
                    tem_onsets(idx_epoch) + para_epoch_leng, :, idx_trial);
            end

            % baseline correction
            tem_epoch = tem_epoch - mean(tem_epoch(para_baseline, :, :));

            % avg across epoch
            tem_epoch = mean(tem_epoch, 3);
            tem_epoch_zscore = zscore(tem_epoch, 0, 1);

            if para_draw_stim
                subplot(3, 1, 2);
                TLab_show_seeg_waveform(tem_epoch(:, tem_stim_ch_idx), para_fs, string(label(tem_stim_ch_idx)));
            end

            % find receiver had significant CCEP
            tem_receiver_ch_idx = [];
            tem_amp_epoch1 = [];
            tem_amp_epoch2 = [];
            for idx_receiver = 1:size(data_sum, 2)
                if idx_receiver == tem_stim_ch_idx % ingore stimulator channel

                    % check trial had < 6 sd or > 6 sd from baseline
                    baseline_std = std(tem_epoch([0.30 * para_fs + 1 : 0.48 * para_fs], idx_receiver));
                    baseline_mean = mean(tem_epoch([0.30 * para_fs + 1 : 0.48 * para_fs], idx_receiver));

                    epoch_1 = tem_epoch([0.51 * para_fs + 1 : 0.53 * para_fs], idx_receiver);
                    epoch_2 = tem_epoch([0.58 * para_fs + 1 : 0.75 * para_fs], idx_receiver);

                    % amplitude
                    epoch_1_amp = max(abs(epoch_1 - baseline_mean));
                    tem_amp_epoch1 = [tem_amp_epoch1 epoch_1_amp];

                    epoch_2_amp = max(abs(epoch_2 - baseline_mean));
                    tem_amp_epoch2 = [tem_amp_epoch2 epoch_2_amp];

                    continue;
                end

                % check trial had < 6 sd or > 6 sd from baseline
                baseline_std = std(tem_epoch([0.30 * para_fs + 1 : 0.48 * para_fs], idx_receiver));
                baseline_mean = mean(tem_epoch([0.30 * para_fs + 1 : 0.48 * para_fs], idx_receiver));

                epoch_1 = tem_epoch([0.51 * para_fs + 1 : 0.53 * para_fs], idx_receiver);
                epoch_2 = tem_epoch([0.58 * para_fs + 1 : 0.75 * para_fs], idx_receiver);

                % amplitude
                epoch_1_amp = max(abs(epoch_1 - baseline_mean));
                tem_amp_epoch1 = [tem_amp_epoch1 epoch_1_amp];

                epoch_2_amp = max(abs(epoch_2 - baseline_mean));
                tem_amp_epoch2 = [tem_amp_epoch2 epoch_2_amp];

                % condition
                condition1_positive = any(epoch_1 > 6 * baseline_std);
                condition1_negative = any(epoch_1 < -6 * baseline_std);

                condition2_positive = any(epoch_2 > 6 * baseline_std);
                condition2_negative = any(epoch_2 < -6 * baseline_std);

                if (condition1_negative && condition2_negative) || (condition1_positive && condition2_positive)
                    tem_receiver_ch_idx = [tem_receiver_ch_idx idx_receiver];
                end
                clear baseline_mean
            end

            if para_draw_stim
                subplot(3, 1, 3);
                TLab_show_seeg_waveform(tem_epoch_zscore(:, tem_receiver_ch_idx), para_fs, label(tem_receiver_ch_idx));
            end

            if ~isempty(tem_receiver_ch_idx)
                save_stim_index = tem_stim_ch_idx;
                save_receiver_index = tem_receiver_ch_idx;
                save_stim_data = tem_epoch(:, save_stim_index);
                save_receiver_data = tem_epoch(:, save_receiver_index);
                save_data = tem_epoch;

                tem_stim_pure =  regexprep(tem_stim_name, '\d+$', '');
                tem_receiver_stim_idx = find(strcmp(regexprep(label(save_receiver_index), '\d+$', ''), tem_stim_pure));
                save_receiver_index(tem_receiver_stim_idx) = [];
                save_receiver_data(:, tem_receiver_stim_idx) = [];

                % amplitude
                save_amp_epoch1 = tem_amp_epoch1';
                save_amp_epoch2 = tem_amp_epoch2';

                % save data
                pathSaving = ['../data/ccep_data/sub-' num2str(idx_subj)];
                if ~exist(pathSaving, 'dir')
                    mkdir(pathSaving);
                else
                    warning('file already exists');
                end

                savename = ['sub-' num2str(idx_subj) '-ses-' num2str(idx_session) '-trial-' num2str(idx_trial) '.mat'];
                save(fullfile(pathSaving, savename),...
                    'save_*', 'label', 'fs');
            end
            clear tem* save_*
        end
        disp(['COMPLETE: subj_' num2str(idx_subj) '_ses_' num2str(idx_session)]);
    end
end