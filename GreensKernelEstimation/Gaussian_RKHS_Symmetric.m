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

l2norm =@(X) sqrt(h * sum(X.^2, 1));

% regularization
lambda = 1e-15;

% variances of x and y Gaussian kernels
sigma = 5e-2;

% Gaussian kernels
G = 1/(sqrt(2*pi)*sigma)*exp(-(x - x').^2/(2*sigma^2));

RK = kron(G, G);

% wavenumber
w = 20;

% Dirichlet boundary conditions
a = 0;
b = 0;

% noise level
sigmanoise = 0;

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
noise = sigmanoise * randn(m, trainsize);
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
us_test = us_test + sigmanoise * randn(m, testsize);

%plot(x, us_train(:, 1))
%hold on
%plot(x, fs_train(:, 1))

% compute the necessary inner product matrices to invert
fs_train_ave = mean(fs_train, 2);
us_train_ave = mean(us_train, 2);

fs_train_centered = fs_train - repmat(fs_train_ave, [1, trainsize]);
us_train_centered = us_train - repmat(us_train_ave, [1, trainsize]);

% compute the optimal coefficients cs through gradient descent
c0 = zeros(m, trainsize);
c = c0;
iters = 10000;
etas = zeros(1, iters);
etas(1:end) = 0.6;

train_errors = zeros(1, iters);
gamma = 0.9;
v_prev = zeros(m, trainsize);
for t = 1:iters
    t
    grad = symm_kron_deriv(c - gamma*v_prev,  fs_train_centered, us_train_centered, G, G, h, lambda);
    v = gamma*v_prev + etas(t)*grad;
    c = c - v;
    
    train_errors(t) = symm_kron_cost(c, fs_train_centered, us_train_centered, G, G, h, lambda);
    v_prev = v;
end

figure(1)
plot(1:iters, train_errors)
title('Training Errors')


% use coefficients to compute Green's kernel
%G_hat = h^2 * (G * c) * (G * fs_train_centered)';
G_hat = h^2 * ((G * c) * (G * fs_train_centered)' + (G * fs_train_centered) * (G * c)') / 2;

%cf = sum(krondim(c, fs_train_centered), 2);
%G_hat = h^2 * reshape((RK*cf)', [m, m])';


% get homogeneous solution
nu = us_train_ave - h * G_hat * fs_train_ave;

figure(2)
plot(x, nu)
title('Homogeneous Solution')

figure(3)
imagesc(G_hat)
title('Predicted Green''s Kernel')

% true Green's kernel
green_modes = 100;
G_true = zeros(m, m);
for j = 1:green_modes
    G_true = G_true + 2 * sin(pi*j*X) .* sin(pi*j*Y) / (pi^2 * j^2 - w^2);
end
figure(4)
imagesc(G_true)
title('True Green''s Kernel')

% compute the relative error of the predicted Green's kernel
kernel_error = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro')

% learned operator is Psi_hat(f) = nu + G_hat * f
% compute the mean relative train error
us_train_hat = repmat(nu, [1, trainsize]) + h * G_hat * fs_train;
train_error = mean(l2norm(us_train_hat - us_train) ./ l2norm(us_train))

% compute the mean relative test error
us_test_hat = repmat(nu, [1, testsize]) + h * G_hat * fs_test;
test_error = mean(l2norm(us_test_hat - us_test) ./ l2norm(us_test))

% plot band of residuals
figure(5)
plot(x, std(us_test_hat - us_test, 0, 2))
title('Standard Deviation of Residuals')
