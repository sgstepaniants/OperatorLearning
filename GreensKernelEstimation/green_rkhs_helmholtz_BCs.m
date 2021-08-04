clear all; close all; clc

addpath('dst_idst')

% Solving Helmholtz equation -u_xx - k^2*u = f with [a, b] Dirichlet BCs

% number of samples
trainsize = 1000;
testsize = 1000;

% number of grid points
m = 500;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

% reproducing kernel for the Sobolev space of Green's kernels
bernoulli2 =@(x) x.^2 - x + 1/6;
bernoulli4 =@(x) x.^4 - 2*x.^3 + x.^2 - 1/30;
[X, Y] = meshgrid(x, x);
RK = bernoulli2(X) .* bernoulli2(Y) / 4 - bernoulli4(abs(X - Y)) / 24;

% functions that are in the null space of the seminorm
Xi = [ones(m, 1), x'];
r = size(Xi, 2);

% regularization
lambda = 1e-5;

% wavenumber
k = 0;

% Dirichlet boundary conditions
a = 1;
b = -1;

% noise level
sigma = 0; %5*1e-2;

% generate train and test input (forcing) functions as Brownian bridges
modes = 100;
ks = (1:modes) / L;
fs_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, trainsize);
fs_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, testsize);

K = spdiags([-ones(m-2, 1), 2*ones(m-2, 1), -ones(m-2, 1)], -1:1, m-2, m-2);
K = full(K)/h^2 - k^2*eye(m-2);

% compute corresponding train and test solutions to Poisson equation
%lambdas = 2 - 2*cos(pi*(1:m-2)/(m-1))';
us_train = zeros(m, trainsize);
for k = 1:trainsize
    f = fs_train(2:m-1, k);
    f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
    %f_hat = dstn(f_pert);
    %u_hat = h^2 * f_hat ./ lambdas;
    %us_train(:, k) = [a; idstn(u_hat); b];
    us_train(:, k) = [a; K\f_pert; b];
end
noise = sigma * randn(m, trainsize);
us_train = us_train + noise;
%snr = db2mag(snr(us_train, noise))

us_test = zeros(m, testsize);
for k = 1:testsize
    f = fs_test(2:m-1, k);
    f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
    %f_hat = dstn(f_pert);
    %u_hat = h^2 * f_hat ./ lambdas;
    %us_test(:, k) = [a; idstn(u_hat); b];
    us_test(:, k) = [a; K\f_pert; b];
end

%plot(x, us_train(:, 1))
%hold on
%plot(x, fs_train(:, 1))

% compute the necessary inner product matrices to invert
fs_train_ave = mean(fs_train, 2);
us_train_ave = mean(us_train, 2);

fs_train_centered = fs_train - repmat(fs_train_ave, [1, trainsize]);
us_train_centered = us_train - repmat(us_train_ave, [1, trainsize]);

%Gamma_ff = h * (fs_train_centered' * fs_train_centered);
Gamma_ff = h * (fs_train_centered' * RK * fs_train_centered);
M = Gamma_ff + trainsize * lambda * eye(trainsize);

%G_hat = us_train_centered * (M \ fs_train_centered');

% solve for the Green's kernel
T = h * fs_train_centered' * Xi;
[Q, R] = qr(T);
Q1 = Q(:, 1:r);
Q2 = Q(:, r+1:end);
R = R(1:r, 1:r);
c = Q2 * ((Q2' * M * Q2) \ (Q2' * us_train_centered'));
d = R \ (Q1' * (us_train_centered' - M * c));
G_hat = c' * fs_train_centered' * RK + d' * Xi';

% get homogeneous solution
nu = us_train_ave - h * G_hat * fs_train_ave;

figure(1)
imagesc(G_hat)
title('Predicted Green''s Kernel')

figure(2)
plot(x, nu)
title('Homogeneous Solution')

% true Green's kernel
%[X, Y] = meshgrid(x, x);
%G_true = cos(k*abs(X - Y))/(2*k);
%figure(2)
%imagesc(G_true)
%title('True Green''s Kernel')

% compute the relative error of the predicted Green's kernel
%kernel_error = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro')

% learned operator is Psi_hat(g) = nu + (us_train - nu) * F_reg \ <f_i, g>
% compute the mean relative train error
us_train_hat = repmat(nu, [1, trainsize]) + h * G_hat * fs_train;
l2norm =@(X) sqrt(sum(X.^2, 1));
train_error = mean(l2norm(us_train_hat - us_train) ./ l2norm(us_train))

% compute the mean relative test error
us_test_hat = repmat(nu, [1, testsize]) + h * G_hat * fs_test;
test_error = mean(l2norm(us_test_hat - us_test) ./ l2norm(us_test))
