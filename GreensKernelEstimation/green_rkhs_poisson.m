clear all; close all; clc

addpath('dst_idst')

% Solving Poisson's equation -u_xx = f with zero Dirichlet BCs

% number of samples
trainsize = 100;
testsize = 1000;
% number of grid points
m = 500;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

% regularizer
lambda = 0;

% noise level
sigma = 0;

% generate train and test input (forcing) functions as Brownian bridges
modes = 100;
ks = (1:modes) / L;
%fs_train = sin(pi*x'*(1:trainsize)/L);
fs_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, trainsize);
fs_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, testsize);

% compute corresponding train and test solutions to Poisson equation
lambdas = 2 - 2*cos(pi*(1:m-2)/(m-1))';
us_train = zeros(m, trainsize);
for k = 1:trainsize
    f = fs_train(2:m-1, k);
    f_hat = dstn(f);
    u_hat = h^2 * f_hat ./ lambdas;
    us_train(:, k) = [0; idstn(u_hat); 0];
end
noise = sigma * randn(m, trainsize);
us_train = us_train + noise;
snr = db2mag(snr(us_train, noise))

us_test = zeros(m, testsize);
for k = 1:testsize
    f = fs_test(2:m-1, k);
    f_hat = dstn(f);
    u_hat = h^2 * f_hat ./ lambdas;
    us_test(:, k) = [0; idstn(u_hat); 0];
end
us_test = us_test;% + sigma * randn(m, testsize);

%plot(x, us_train(:, 1))
%hold on
%plot(x, fs_train(:, 1))

% learn the Green's kernel of this Poisson BVP
F = h * (fs_train' * fs_train);
F_reg = F + trainsize * lambda * eye(trainsize);
G_hat = us_train * (F_reg \ fs_train');
figure(1)
imagesc(G_hat)
title('Predicted Green''s Kernel')

% bootsrap to learn the Green's kernel
batchsize = 10;
iters = 100 * round(trainsize / batchsize);
%G_batches = zeros(m, m, iters);
G_bootstrap = zeros(m, m);
for k = 1:iters
    batch_inds = randperm(trainsize, batchsize);
    %batch_inds = randsample(trainsize, batchsize, true);
    fs_train_batch = fs_train(:, batch_inds);
    us_train_batch = us_train(:, batch_inds);
    %G_batches(:, :, k) = us_train_batch * (F_reg(batch_inds, batch_inds) \ fs_train_batch');
    G_bootstrap = G_bootstrap + us_train_batch * ((F(batch_inds, batch_inds) + lambda * eye(batchsize)) \ fs_train_batch');
end
G_bootstrap = G_bootstrap / iters;
%G_bootstrap = mean(G_batches, 3);
figure(2)
imagesc(G_bootstrap)
title('Bootstrapped Green''s Kernel')

% average samples to learn the Green's kernel
subsets = round(sqrt(trainsize));
w = ceil(trainsize / subsets);
fs_train_ave = zeros(m, subsets);
us_train_ave = zeros(m, subsets);
for k = 1:subsets
    fs_train_ave(:, k) = mean(fs_train(:, 1+(k-1)*w:min(k*w, trainsize)), 2);
    us_train_ave(:, k) = mean(us_train(:, 1+(k-1)*w:min(k*w, trainsize)), 2);
end
F_ave = h * (fs_train_ave' * fs_train_ave);
F_reg_ave = F_ave + lambda * eye(subsets);
G_ave = us_train_ave * (F_reg_ave \ fs_train_ave');
figure(3)
imagesc(G_ave)
title('Window Average Green''s Kernel')
G_hat = G_ave;

% compute OLS Green's kernel using functional principal component regression
gamma_ff = cov(fs_train', 1);
r = 5; % largest mode to keep in gamma_ff

[phi_train, lambda_train] = eig(gamma_ff);
[~, ind] = sort(diag(lambda_train), 'descend');
lambda_train = lambda_train(ind, ind);
phi_train = phi_train(:, ind);

gamma_uf = crosscov(us_train', fs_train');
phi_train_trunc = phi_train(:, 1:r);
lambda_train_trunc = lambda_train(1:r, 1:r);
G_pcr = gamma_uf * phi_train_trunc * pinv(lambda_train_trunc) * phi_train_trunc';
figure(4)
imagesc(G_pcr)
title('Principal Component Regression Green''s Kernel')
G_hat = G_pcr;

% true Green's kernel
[X, Y] = meshgrid(x, x);
G_true = (X + Y - abs(Y - X))/2 - X.*Y;
figure(5)
imagesc(G_true)
title('True Green''s Kernel')

%10^(snr(G_hat, G_hat - G_true) / 20)

% compute the relative error of the predicted Green's kernel
kernel_error = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro')

% learned operator is Psi_hat(g) = us_train * F_reg \ <f_i, g>
% compute the mean relative train error
us_train_hat = h * G_hat * fs_train;
l2norm =@(X) sqrt(sum(X.^2, 1));
train_error = mean(l2norm(us_train_hat - us_train) ./ l2norm(us_train))

% compute the mean relative test error
us_test_hat = h * G_hat * fs_test;
test_error = mean(l2norm(us_test_hat - us_test) ./ l2norm(us_test))
