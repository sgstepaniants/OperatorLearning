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

% functions that are in the null space of the seminorm
r = 1;
Xi = ones(m, m);

%r = 3;
%Xi = zeros(m, m, r);
%Xi(:, :, 1) = ones(m, m);
%Xi(:, :, 2) = X;
%Xi(:, :, 3) = Y;

% regularization
lambda = 1e-6;

% number of modes of reproducing kernel to keep
RKmodes = m-2; % Beware of Nyquist frequency! Don't let this exceed m.

% wavenumber
w = 0;

% Dirichlet boundary conditions
a = 0;
b = 0;

% noise level
sigma = 1e-2;

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

%plot(x, us_train(:, 1))
%hold on
%plot(x, fs_train(:, 1))

% compute the necessary inner product matrices to invert
fs_train_ave = mean(fs_train, 2);
us_train_ave = mean(us_train, 2);

fs_train_centered = fs_train - repmat(fs_train_ave, [1, trainsize]);
us_train_centered = us_train - repmat(us_train_ave, [1, trainsize]);

% compute T
T = zeros(m, trainsize, r);
for k = 1:r
    T(:, :, k) = h * Xi(:, :, k) * fs_train_centered;
end

% solve for the eigenfunctions of M
Vk = zeros(trainsize, trainsize, RKmodes);
Dk = zeros(trainsize, trainsize, RKmodes);
for k = 1:RKmodes
    Mk = zeros(trainsize, trainsize);
    for l = 1:RKmodes
        vec = h * sin(pi * l * x) * fs_train_centered;
        Mk = Mk + 2 * (vec' * vec) / (pi^2 * (k^2 + l^2));
    end
    [Vk(:, :, k), Dk(:, :, k)] = eig(Mk);
end

% compute the inverse of M applied to us and Ts
Minv_u = zeros(m, trainsize);
Minv_T = zeros(m, trainsize, r);
orth_u = us_train_centered;
orth_T = T;
for k = 1:RKmodes
    eigsk = diag(Dk(:, :, k));
    numeigsk = length(eigsk);
    for l = 1:numeigsk
        eigfun = sin(pi * k * x)' * Vk(:, l, k)';
        eigfun = eigfun / sqrt(h * sum(sum(eigfun.^2)));
        proj_u = h * sum(sum(us_train_centered .* eigfun)) * eigfun;
        orth_u = orth_u - proj_u;
        Minv_u = Minv_u + 1 / (eigsk(l) + trainsize * lambda) * proj_u;
        for j = 1:r
            proj_T = h * sum(sum(T(:, :, j) .* eigfun)) * eigfun;
            orth_T(:, :, j) = orth_T - proj_T;
            Minv_T(:, :, j) = Minv_T(:, :, j) + 1 / (eigsk(l) + trainsize * lambda) * proj_T;
        end
    end
end
Minv_u = Minv_u + 1 / (trainsize * lambda) * orth_u;
Minv_T = Minv_T + 1 / (trainsize * lambda) * orth_T;


% compute c and d coefficients
A = zeros(r, r);
b = zeros(r, 1);
for j = 1:r
    for k = 1:r
        A(j, k) = h * sum(sum(T(:, :, j) .* Minv_T(:, :, k)));
    end
    b(j) = h * sum(sum(T(:, :, j) .* Minv_u));
end
d = A \ b;
c = Minv_u - reshape(squeeze(reshape(Minv_T, [m*trainsize, r]) * d), [m, trainsize]);

% use coefficients to compute Green's kernel
Kc = zeros(m, m);
for k = 1:RKmodes
    for l = 1:RKmodes
        val = 0;
        for j = 1:trainsize
            val = val + h^2 * sum(sin(pi*k*x)' .* c(:, j)) * sum(sin(pi*l*x)' .* fs_train_centered(:, j));
        end
        Kc = Kc + 4 * sin(pi*k*x)' * sin(pi*l*x) * val / (pi^2*(k^2+l^2));
    end
end
G_hat = Kc + reshape(reshape(Xi, [m^2, r]) * d, [m, m]);


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
