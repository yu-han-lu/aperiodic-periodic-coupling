%%
clear; clc; close all; warning off;
addpath('../../tools/');
addpath('/home/hangze/Downloads/fieldtrip-20250106/');

%%
% input data
data = load('../data/fig1c_oscillation_timedomain.mat');
x = data.x_all;
fs = 1000;
clear data

% irasa
T = size(x, 2)./fs;
win_length = 30;
f_range = [0.01 400];
[f, original_powspectra, fractal_powspectra, oscillatory_powspectra] = ...
    IRASA_hz(fs, T, x, win_length, f_range);

% save data
save_path = '../data/fig1c_irasa_data.mat';
save(save_path, 'f', 'original_powspectra', 'fractal_powspectra', 'oscillatory_powspectra', '-v7.3');