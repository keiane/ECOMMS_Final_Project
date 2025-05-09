% === MAINV2.M ===
% Complete script for RIS-based DOA estimation using Atomic Norm Minimization with snapshot averaging and FFT comparison

clear all; close all; clc;

rng(111);  % Ensure reproducibility to match original figures

% === Parameters and Helpers ===
param.vec = @(MAT) MAT(:);
param.vecH = @(MAT) MAT(:).';

param.theta_RS = 0;
param.d_E = 0.5;
param.M = 64;               % Number of RIS elements
param.N = 16;               % Number of snapshots
param.K = 3;                % Number of vehicles
SNR_dB = 20;

% === Define steering vector generator ===
param.get_steer = @(theta, L) exp(1j*2*pi*[0:1:L-1].' * param.d_E * param.vecH(sind(theta + param.theta_RS)));

% === Geometry and angles ===
param.theta_AR = -10;                      % Direction AP → RIS (fixed for match)
param.theta_TR = [-25; 15; 30];            % True DOAs (fixed for visual match)

% === Distances ===
param.d_AT = 20; param.d_TR = 30; param.d_RS = 3; param.d_AR = 5;

% === Measurement matrix optimization ===
a_tmp = param.get_steer(param.theta_AR, param.M);
A_tmp = a_tmp * a_tmp';
cvx_begin sdp quiet
    variable G_tilde(param.M, param.M) hermitian
    minimize(trace(A_tmp * G_tilde))
    subject to
        G_tilde >= 0
        diag(G_tilde) == 1
cvx_end

[U_tilde, D_tilde] = eig(G_tilde);
D_tilde = diag(sqrt(real(diag(D_tilde))));
g_tilde = sqrt(1/2) * (randn(param.M, 1e4) + 1j * randn(param.M, 1e4));
G_tmp = exp(1j * angle(U_tilde * D_tilde * g_tilde));
G = G_tmp';
param.G = G(1:param.N, :);

% === Snapshot Generation ===
R_all = zeros(param.N, param.N);
for snap_idx = 1:param.N
    z = 1/(param.d_AT*param.d_TR*param.d_RS) * exp(1j * rand(param.K, 1));
    q = 1/(param.d_AR*param.d_RS) * exp(1j * rand(1));

    r_z = param.G * param.get_steer(param.theta_TR, param.M) * z;
    r_q = param.G * param.get_steer(param.theta_AR, param.M) * q;

    noise_ref = r_z;
    noise_pow = norm(noise_ref)^2 / 10^(SNR_dB/10);
    w = sqrt(noise_pow) * (randn(size(r_z)) + 1j * randn(size(r_z))) / norm(randn(size(r_z)) + 1j * randn(size(r_z)));

    R_all(:, snap_idx) = r_z + r_q + w;
end

r_mean = mean(R_all, 2);
b = param.G * param.get_steer(param.theta_AR, param.M);

% === Atomic Norm Minimization (ANM) ===
rho = sqrt(log(param.M) * param.M) * sqrt(noise_pow);
cvx_begin sdp quiet
    variable est_xi(param.M, 1) complex
    variable u(param.M, 1) complex
    variable est_eta complex
    variable Z(param.M, param.M) hermitian toeplitz
    variable nu(1,1)
    minimize(quad_form(r_mean - param.G * est_xi - est_eta * b, eye(length(r_mean))) + ...
             rho/2 * (nu + (1/param.M) * trace(Z)))
    subject to
        [Z, est_xi; est_xi', nu] >= 0
        Z(:, 1) == u
cvx_end

% === MUSIC via Hankel Matrix ===
param.cont_ang = (-45:0.001:45)';
est_x = MUSIConesnapshot(est_xi, param);
est_x = est_x(:);

N_plot = min(length(param.cont_ang), length(est_x));
est_spectrum = [param.cont_ang(1:N_plot), abs(est_x(1:N_plot))];
rmse_propose = get_rmse(est_spectrum, param.theta_TR);

fprintf("RMSE (Proposed Method): %.4f degrees\n", rmse_propose);

% === CRLB Calculation ===
B = zeros(param.M, param.K);
for idx = 1:param.K
    steer = param.get_steer(param.theta_TR(idx), param.M);
    B(:, idx) = 1j * 2 * pi * param.d_E * z(idx) * cosd(param.theta_TR(idx) + param.theta_RS) .* (steer .* (0:param.M-1).');
end
G = param.G;
S1 = [B' * G' * G * B, B' * G' * G * param.get_steer(param.theta_TR, param.M), B' * G' * G * param.get_steer(param.theta_AR, param.M)];
S2 = [param.get_steer(param.theta_TR, param.M)' * G' * G * B, param.get_steer(param.theta_TR, param.M)' * G' * G * param.get_steer(param.theta_TR, param.M), param.get_steer(param.theta_TR, param.M)' * G' * G * param.get_steer(param.theta_AR, param.M)];
S3 = [param.get_steer(param.theta_AR, param.M)' * G' * G * B, param.get_steer(param.theta_AR, param.M)' * G' * G * param.get_steer(param.theta_TR, param.M), param.get_steer(param.theta_AR, param.M)' * G' * G * param.get_steer(param.theta_AR, param.M)];
F = length(r_mean)/noise_pow * [S1; S2; S3];
crlb_all = abs(diag(inv(F)));
crlb_deg = rad2deg(sqrt(sum(crlb_all(1:param.K))/param.K));

fprintf("CRLB (Theoretical Lower Bound): %.4f degrees\n", crlb_deg);

% === FFT-Based DOA Estimation ===
fft_angles = -45:0.5:45;
steering_fft = param.get_steer(fft_angles, param.M);
r_fft = r_mean;
fft_spectrum = abs(steering_fft' * (param.G' * r_fft)).^2;
fft_spectrum = 10*log10(fft_spectrum / max(fft_spectrum));

% === Standard Paper-style Visualizations ===
figure;
tiledlayout(1,3);

% --- Panel 1: Beamforming matrix response ---
nexttile;
ang_range = -50:0.1:50;
pow_g = abs(param.G * param.get_steer(ang_range, param.M)).^2;
pow_g = mean(pow_g, 1);
pow_g = 10 * log10(pow_g / max(pow_g));
plot(ang_range, pow_g, 'b', 'LineWidth', 2); hold on;
stem(param.theta_AR, -7, 'r', 'LineWidth', 2);
title('Optimized measurement matrix');
xlabel('Spatial angle (deg)'); ylabel('Beamforming power (dB)');
grid on; legend('Measurement matrix', 'AP direction');

% --- Panel 2: FFT-based DOA estimation ---
nexttile;
plot(fft_angles, fft_spectrum, 'b', 'LineWidth', 2); hold on;
stem(param.theta_TR, min(fft_spectrum) * ones(size(param.theta_TR)), 'r', 'LineWidth', 2);
title('FFT-based DOA Estimation');
xlabel('Angle (degrees)'); ylabel('FFT Spectrum (dB)');
grid on; legend('FFT spectrum', 'True DOAs');

% --- Panel 3: Proposed Method (ANM+MUSIC) ---
nexttile;
sp_db = 20*log10(abs(est_x));
sp_db = sp_db - max(sp_db);
plot(param.cont_ang(1:N_plot), sp_db(1:N_plot), 'b', 'LineWidth', 2); hold on;
stem(param.theta_TR, max(sp_db) * ones(size(param.theta_TR)), 'r', 'LineWidth', 2);
title('Proposed method');
xlabel('Angle (degrees)'); ylabel('Spatial Spectrum (dB)');
grid on; legend('Proposed method', 'True DOAs');
