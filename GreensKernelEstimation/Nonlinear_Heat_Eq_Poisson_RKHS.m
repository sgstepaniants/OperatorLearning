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

% functions that are in the null space of the seminorm
r = 1;
Xi = ones(m, m);

% regularization
lambda = 1e-5;

% number of modes of reproducing kernel to keep
RKmodes = m-2; % Beware of Nyquist frequency! Don't let this exceed m.

% noise level
sigma = 0;

% generate train and test input (forcing) functions as Brownian bridges
datamodes = 100;
ks = (1:datamodes) / L;
fs = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(datamodes, fullsize);
fs_train = fs(:, 1:trainsize);
fs_test = fs(:, trainsize+1:end);

% compute corresponding train and test solutions to Poisson equation
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
G_true = 1/(2*sqrt(pi*t(tpred)))*exp(-(X - Y).^2/(4*t(tpred)));
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

ind = 1;
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
