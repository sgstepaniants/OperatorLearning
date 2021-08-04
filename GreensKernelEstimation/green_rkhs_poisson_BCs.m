clear all; close all; clc

addpath('dst_idst')

% Solving Poisson's equation -u_xx = f with [a, b] Dirichlet BCs

% number of samples
trainsize = 200;
testsize = 1000;
% number of grid points
m = 500;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

% Dirichlet boundary conditions
a = 1;
b = -1;

% regularizer
lambda = 0;

% generate train and test input (forcing) functions as Brownian bridges
modes = 100;
ks = (1:modes) / L;
fs_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, trainsize);
fs_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, testsize);

K = spdiags([-ones(m-2, 1), 2*ones(m-2, 1), -ones(m-2, 1)], -1:1, m-2, m-2);
K = full(K) / h^2;

% compute corresponding train and test solutions to Poisson equation
lambdas = 2 - 2*cos(pi*(1:m-2)/(m-1))';
us_train = zeros(m, trainsize);
for k = 1:trainsize
    f = fs_train(2:m-1, k);
    f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
    %f_hat = dstn(f_pert);
    %u_hat = h^2 * f_hat ./ lambdas;
    %us_train(:, k) = [a; idstn(u_hat); b];
    us_train(:, k) = [a; K\f_pert; b];
end

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

% learn the Green's kernel of this Poisson BVP
traininds1 = 1:trainsize/2;
traininds2 = trainsize/2+1:trainsize;
trainsize1 = length(traininds1);
trainsize2 = length(traininds2);
fs_train1 = fs_train(:, traininds1);
fs_train2 = fs_train(:, traininds2);
us_train1 = us_train(:, traininds1);
us_train2 = us_train(:, traininds2);

F = h * (fs_train1' * fs_train1);
%F = diag(ones(1, trainsize1)/6);
F_reg = F + lambda * eye(trainsize1);

inprods = h * (fs_train1' * fs_train2);
cs = sum(F_reg \ inprods, 1)';
normalizer = sum((1 - cs).^2);
nu = (us_train2 - us_train1 * (F_reg \ inprods)) * (1 - cs);
nu = nu / normalizer;

G_hat = (us_train1 - repmat(nu, [1, trainsize1])) * (F_reg \ fs_train1');
figure(1)
imagesc(G_hat)
title('Predicted Green''s Kernel')

% true Green's kernel
[X, Y] = meshgrid(x, x);
G_true = (X + Y - abs(Y - X))/2 - X.*Y;
figure(2)
imagesc(G_true)
title('True Green''s Kernel')

figure(3)
plot(x, nu)
title('Solution to Homogeneous Equation')

% compute the relative error of the predicted Green's kernel
kernel_error = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro')

% learned operator is Psi_hat(g) = nu + (us_train - nu) * F_reg \ <f_i, g>
% compute the mean relative train error
inprods = h * (fs_train1' * fs_train);
us_train_hat = repmat(nu, [1, trainsize]) + (us_train1 - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
l2norm =@(X) sqrt(sum(X.^2, 1));
train_error = mean(l2norm(us_train_hat - us_train) ./ l2norm(us_train))

% compute the mean relative test error
inprods = h * (fs_train1' * fs_test);
us_test_hat = repmat(nu, [1, testsize]) + (us_train1 - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
test_error = mean(l2norm(us_test_hat - us_test) ./ l2norm(us_test))
