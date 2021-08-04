clear all; close all; clc


% Solving wave equation u_tt - c^2*u_xx = 0 with zero Dirichlet BC's and zero velocities.
% Learn map from initial wave to wave at future time.

% number of samples
trainsize = 200;
testsize = 1000;

% number of grid points
m = 500;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

% number of time points
n = 500;
T = 1;
dt = T/(n-1);
t = linspace(0, T, n);

% wave speed
c = 1;

% regularizer
lambda = 0;

% generate train and test input (forcing) functions as Brownian bridges
modes = 100;
ks = (1:modes) / L;
ps_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, trainsize);
ps_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, testsize);
vs_train = zeros(m, trainsize);
vs_test = zeros(m, testsize);

pulse = sin(20*x);
%pulse(x < 0) = 0;
pulse(x > pi/20) = 0;

% compute corresponding train and test solutions to Poisson equation
us_train = zeros(m, n, trainsize);
for k = 1:trainsize
    us_train(:, :, k) = wave_eq(ps_train(:, k), vs_train(:, k), c, x, t);
end

us_test = zeros(m, n, testsize);
for k = 1:testsize
    us_test(:, :, k) = wave_eq(ps_test(:, k), vs_test(:, k), c, x, t);
end

%plot(x, us_train(:, 10, 1))
%hold on
%plot(x, ps_train(:, 1))

% learn the Green's kernel of the wave equation
trainsize1 = trainsize/2;
trainsize2 = trainsize - trainsize1;
ps_train1 = ps_train(:, 1:trainsize1);
ps_train2 = ps_train(:, trainsize1+1:end);

tmpt = 4*n/5;
us_train_tmpt = squeeze(us_train(:, tmpt, :));
us_test_tmpt = squeeze(us_test(:, tmpt, :));
us_train1_tmpt = us_train_tmpt(:, 1:trainsize1);
us_train2_tmpt = us_train_tmpt(:, trainsize1+1:end);

F = h * (ps_train1' * ps_train1);
F_reg = F + lambda * eye(trainsize1);

inprods = h * (ps_train1' * ps_train2);
cs = sum(F_reg \ inprods, 1)';
normalizer = sum((1 - cs).^2);
nu = (us_train2_tmpt - us_train1_tmpt * (F_reg \ inprods)) * (1 - cs);
nu = nu / normalizer;

G_pos_hat = (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ ps_train1');
figure(1)
imagesc(x, x, G_pos_hat)
title('Predicted Green''s Kernel')
set(gca,'YDir','normal')

% true Green's kernel
modes = 100;
G_pos_true = zeros(m, m);
for k = 1:modes
    G_pos_true = G_pos_true + 2/L * sin(k*pi*x/L)' * sin(k*pi*x/L) * cos(k*pi*c*t(tmpt)/L);
end
%G_pos_true = zeros(m, m, n);
%for j = 1:m
%    j
%    for k = 1:modes
%        G_pos_true(:, :, j) = G_pos_true(:, :, j) + 2/L * sin(k*pi*x/L)' * sin(k*pi*x/L) * cos(k*pi*c*t(j)/L);
%    end
%end

[X, Y] = meshgrid(x, x);
figure(2)
imagesc(x, x, G_pos_true)
title('True Green''s Kernel')
set(gca,'YDir','normal')

figure(3)
plot(x, nu)
title('Solution to Homogeneous Equation')

% compute the relative error of the predicted Green's kernel
kernel_error = norm(G_pos_hat - G_pos_true, 'fro') / norm(G_pos_true, 'fro')

% learned operator is Psi_hat(g) = nu + (us_train - nu) * F_reg \ <f_i, g>
% compute the mean relative train error
inprods = h * (ps_train1' * ps_train);
us_train_tmpt_hat = repmat(nu, [1, trainsize]) + (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
l2norm =@(X) sqrt(sum(X.^2, 1));
train_error = mean(l2norm(us_train_tmpt_hat - us_train_tmpt) ./ l2norm(us_train_tmpt))

% compute the mean relative test error
inprods = h * (ps_train1' * ps_test);
us_test_tmpt_hat = repmat(nu, [1, testsize]) + (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
test_error = mean(l2norm(us_test_tmpt_hat - us_test_tmpt) ./ l2norm(us_test_tmpt))



tmpt = 500;
us_train_tmpt = squeeze(us_train(:, tmpt, :));
us_test_tmpt = squeeze(us_test(:, tmpt, :));
us_train1_tmpt = us_train_tmpt(:, 1:trainsize1);
us_train2_tmpt = us_train_tmpt(:, trainsize1+1:end);

F = h * (ps_train1' * ps_train1);
F_reg = F + lambda * eye(trainsize1);

inprods = h * (ps_train1' * ps_train2);
cs = sum(F_reg \ inprods, 1)';
normalizer = sum((1 - cs).^2);
nu = (us_train2_tmpt - us_train1_tmpt * (F_reg \ inprods)) * (1 - cs);
nu = nu / normalizer;

pulse =@(x) sin(20*x) .* (x < pi/20);
inprods = h * (ps_train1' * pulse(x)');
pred_sol = nu + (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);

true_sol = zeros(m, n);
modes = 100;
for j=1:modes
    A = 2/L * h * dot(sin(j*pi*x/L), pulse(x));
    true_sol = true_sol + A * sin(j*pi*x/L)' * cos(j*pi*c*t/L);
    
    B = 2/(j*c*pi) * h * dot(sin(j*pi*x/L), zeros(m, 1));
    true_sol = true_sol + B * sin(j*pi*x/L)' * sin(j*pi*c*t/L);
end

plot(x, pred_sol)
%hold on;
%plot(x, true_sol(:, tmpt))
