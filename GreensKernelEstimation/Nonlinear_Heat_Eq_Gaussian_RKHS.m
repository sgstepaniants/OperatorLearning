clear all; close all; clc

% Solving nonlinear heat equation

% number of samples
trainsize = 100;
testsize = 100;
fullsize = trainsize + testsize;

% number of grid points
m = 100;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

% number of time points
n = 100;
T = 0.2;
dt = T/(n-1);
t = linspace(0, T, n);

% timepoint into the future to predict
tpred = 10;

[X, Y] = meshgrid(x, x);

% regularization
lambda = 1e-10;

% variances of x and y Gaussian kernels
sigma1 = 5e-2;
sigma2 = 5e-2;

% Gaussian kernels
G1 = 1/(sqrt(2*pi)*sigma1)*exp(-(x - x').^2/(2*sigma1^2));
G2 = 1/(sqrt(2*pi)*sigma2)*exp(-(x - x').^2/(2*sigma2^2));

% noise level
sigma = 0;

% generate train and test input (forcing) functions as Brownian bridges
datamodes = 100;
ks = (1:datamodes) / L;
fs = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(datamodes, fullsize);
fs_train = fs(:, 1:trainsize);
fs_test = fs(:, trainsize+1:end);

% compute corresponding train and test solutions to nonlinear heat equation
sols = zeros(m, n, fullsize);
for k = 1:fullsize
    k
    heatic =@(x) fs(max(ceil(m*x/L), 1), k);
    sol = pdepe(0, @heatpde, heatic, @heatbc, x, t);
    sols(:, :, k) = sol';
end
noise = sigma * randn(m, n, fullsize);
sols = sols + noise;
signalnoiseratio = db2mag(snr(sols(:), noise(:)))

us_train = squeeze(sols(:, tpred, 1:trainsize));
us_test = squeeze(sols(:, tpred, trainsize+1:end));

plot(x, us_train(:, 1))
hold on
plot(x, fs_train(:, 1))

% compute the necessary inner product matrices to invert
fs_train_ave = mean(fs_train, 2);
us_train_ave = mean(us_train, 2);

fs_train_centered = fs_train - repmat(fs_train_ave, [1, trainsize]);
us_train_centered = us_train - repmat(us_train_ave, [1, trainsize]);

% compute the Gaussian inner product matrix
Gamma = h^2 * fs_train_centered' * G2 * fs_train_centered;
Gamma = (Gamma + Gamma') / 2;
[Vgamma, Dgamma] = eig(Gamma);
[V1, D1] = eig(G1);
dgamma = diag(Dgamma);
d1 = diag(D1);

% compute the inverse of M applied to us
Minv_u = V1' * us_train_centered * Vgamma;
Minv_u = Minv_u ./ (h * d1 * dgamma' + trainsize*lambda);
Minv_u = V1 * Minv_u * Vgamma';

% use coefficients to compute Green's kernel
G_hat = h^2 * (G1 * Minv_u) * (G2 * fs_train_centered)';

% get homogeneous solution
nu = us_train_ave - h * G_hat * fs_train_ave;

figure(1)
plot(x, nu)
title('Homogeneous Solution')

figure(2)
imagesc(G_hat)
title('Predicted Green''s Kernel')

% % true Green's kernel
% green_modes = 100;
% G_true = zeros(m, m);
% for j = 1:green_modes
%     G_true = G_true + 2 * sin(pi*j*X) .* sin(pi*j*Y) / (pi^2 * j^2 - w^2);
% end
% figure(3)
% imagesc(G_true)
% title('True Green''s Kernel')
% 
% % compute the relative error of the predicted Green's kernel
% kernel_error = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro')

% learned operator is Psi_hat(f) = nu + G_hat * f
% compute the mean relative train error
us_train_hat = repmat(nu, [1, trainsize]) + h * G_hat * fs_train;
l2norm =@(X) sqrt(sum(X.^2, 1));
train_error = mean(l2norm(us_train_hat - us_train) ./ l2norm(us_train))

% compute the mean relative test error
us_test_hat = repmat(nu, [1, testsize]) + h * G_hat * fs_test;
test_error = mean(l2norm(us_test_hat - us_test) ./ l2norm(us_test))

% plot band of residuals
figure(4)
plot(x, std(us_test_hat - us_test, 0, 2))
title('Standard Deviation of Residuals')


ind = 59;
plot(us_test(:, ind))
hold on
plot(us_test_hat(:, ind))
title('Approximation of Solution')
legend('true', 'predicted')

function [c, f, s] = heatpde(x, t, u, dudx)
    alpha = 0;
    c = 1;
    f = exp(alpha*u)*dudx;
    s = 0;
end

function [pl, ql, pr, qr] = heatbc(xl, ul, xr, ur, t)
    pl = ul;
    ql = 0;
    pr = ur;
    qr = 0;
end
