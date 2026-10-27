% This script plots 100-channel EEG data on a single figure.
% Each channel is plotted on the y-axis from top to bottom,
% with no overlap between the signals. The x-axis represents
% the time points for each channel's signal.
%
% Yuhan Lu, Jul 8 2024

function TLab_show_seeg_waveform(data, fs, channel_name)

channels = size(data,2);
% time = [1:size(data,1)]/fs;
time = (-300+1:1:500)./fs;

cmap = parula(256); % hangze

% Create a figure window
% figure;

% Loop through each channel and plot on the same figure
hold on;
tik = 1.5;
for i = 1:channels
    col = cmap(5*(i-1)+1, :); % hangze

    % Offset each channel's signal to avoid overlap
    offset = (i - 1) * tik;  % Adjust the offset as needed
    plot(time, data(:, i) + offset, 'Color', col); % hangze
end
hold off;

% Set y-axis labels to show channel numbers
yticks((0:channels-1) * tik);
yticklabels(channel_name);

% Set axis labels and title
xlabel('Time (s)');

% Optimize the figure display
axis tight;
grid on;

end