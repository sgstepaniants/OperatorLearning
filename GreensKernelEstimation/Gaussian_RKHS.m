clear all; close all; clc

addpath('dst_idst')

% Solving Helmholtz equation -u_xx - w^2*u = f with [a, b] Dirichlet BCs

% number of samples
trainsize = 100;
testsize = 1000;

% number of grid points
m = 100;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

[X, Y] = meshgrid(x, x);

% regularization
lambda = 1e-15;

% variances of x and y Gaussian kernels
sigma1 = 5e-2;
sigma2 = 5e-2;

% Gaussian kernels
G1 = 1/(sqrt(2*pi)*sigma1)*exp(-(x - x').^2/(2*sigma1^2));
G2 = 1/(sqrt(2*pi)*sigma2)*exp(-(x - x').^2/(2*sigma2^2));

% wavenumber
w = 20;

% Dirichlet boundary conditions
a = 0;
b = 0;

% noise level
sigma = 0;

% generate train and test input (forcing) functions as Brownian bridges
datamodes = 100;
ks = (1:datamodes) / L;
fs_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(datamodes, trainsize);
fs_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(datamodes, testsize);

K = spdiags([-ones(m-2, 1), 2*ones(m-2, 1), -ones(m-2, 1)], -1:1, m-2, m-2);
K = full(K)/h^2 - w^2*eye(m-2);

% compute corresponding train and test solutions to Poisson equation
us_train = zeros(m, trainsize);
for k = 1:trainsize
    f = fs_train(2:m-1, k);
    f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
    us_train(:, k) = [a; K\f_pert; b];
end
noise = sigma * randn(m, trainsize);
us_train = us_train + noise;

%us_train_diff = us_train - repmat(1 - 2*x', [1, trainsize]);
%us_train_diff = us_train - repmat(nu, [1, trainsize]);
%signalnoiseratio = db2mag(snr(us_train_diff(:), noise(:)))
signalnoiseratio = db2mag(snr(us_train(:), noise(:)))

us_test = zeros(m, testsize);
for k = 1:testsize
    f = fs_test(2:m-1, k);
    f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
    us_test(:, k) = [a; K\f_pert; b];
end
us_test = us_test + sigma * randn(m, testsize);

%plot(x, us_train(:, 1))
%hold on
%plot(x, fs_train(:, 1))

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

% true Green's kernel
green_modes = 100;
G_true = zeros(m, m);
for j = 1:green_modes
    G_true = G_true + 2 * sin(pi*j*X) .* sin(pi*j*Y) / (pi^2 * j^2 - w^2);
end
figure(3)
imagesc(G_true)
title('True Green''s Kernel')

% compute the relative error of the predicted Green's kernel
kernel_error = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro')

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
