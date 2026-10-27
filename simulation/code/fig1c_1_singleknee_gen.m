%%
clear; clc; close all; warning off;

%% simulate_ou
% parameter
dt = 0.001;
T_list = 30;
knee_list = 5;
tau_list = 1 ./ (knee_list*2*pi);

% noise
Q11 = 1;

% main
T = T_list;
t = 0.001:dt:T;
N = length(t);
tau = tau_list;

w = sqrt(Q11 / dt) * randn(1, N);
x = zeros(1, N);
for k = 1:N-1
    dx = -x(k)/tau + w(k);
    x(k+1) = x(k) + dx * dt;
end

% save data
save_path = '../data/fig1c_OU_data.mat';
save(save_path, 'x', '-v7.3');