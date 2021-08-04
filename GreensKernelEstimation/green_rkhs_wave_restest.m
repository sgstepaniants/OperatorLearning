clear all; close all; clc

% Show the performance of the Green RKHS estimator over varying mesh sizes

% grid resolutions to test
resolutions = 50:50:1000;
numres = length(resolutions);

% regularizer
lambda = 0;

% timepoint to predict wave position at
pred_time = 3/5;

% number of modes to keep in KL expansion of Brownian bridge
modes = 100;

% number of trials to average over
numtrials = 2;

% number of samples
trainsize = 200;
testsize = 1000;

% train dataset split
traininds1 = 1:trainsize/2;
traininds2 = trainsize/2+1:trainsize;
trainsize1 = length(traininds1);
trainsize2 = length(traininds2);

kernel_errors = zeros(numres, numtrials);
train_errors = zeros(numres, numtrials);
test_errors = zeros(numres, numtrials);
ps_trains = cell(numres, numtrials);
ps_tests = cell(numres, numtrials);
us_trains = cell(numres, numtrials);
us_tests = cell(numres, numtrials);
pred_nus = cell(numres, numtrials);
pred_kernels = cell(numres, numtrials);
for j=1:numres
    j
    for s=1:numtrials
        % number of grid points
        m = resolutions(j);
        L = 1;
        h = L/(m-1);
        x = linspace(0, L, m);
        
        % number of time points
        n = m;
        T = 1;
        dt = T/(n-1);
        t = linspace(0, T, n);

        % wave speed
        c = 1;
        
        % generate train and test input (forcing) functions as Brownian bridges
        ks = (1:modes) / L;
        ps_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, trainsize);
        %ps_trains{j, s} = ps_train;
        ps_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, testsize);
        %ps_tests{j, s} = ps_test;
        
        vs_train = ones(m, trainsize);
        vs_test = ones(m, testsize);
        
        % compute corresponding train and test solutions to Poisson equation
        us_train = zeros(m, n, trainsize);
        for k = 1:trainsize
            us_train(:, :, k) = wave_eq(ps_train(:, k), vs_train(:, k), c, x, t);
        end
        %us_trains{j, s} = us_train;
        
        us_test = zeros(m, n, testsize);
        for k = 1:testsize
            us_test(:, :, k) = wave_eq(ps_test(:, k), vs_test(:, k), c, x, t);
        end
        %us_tests{j, s} = us_test;
        'hello'
        
        % split train inputs and outputs into two groups
        ps_train1 = ps_train(:, 1:trainsize1);
        ps_train2 = ps_train(:, trainsize1+1:end);
        
        tmpt = round(pred_time * n);
        us_train_tmpt = squeeze(us_train(:, tmpt, :));
        us_test_tmpt = squeeze(us_test(:, tmpt, :));
        us_train1_tmpt = us_train_tmpt(:, 1:trainsize1);
        us_train2_tmpt = us_train_tmpt(:, trainsize1+1:end);
        
        % learn the Green's kernel of the wave equation
        F = h * (ps_train1' * ps_train1);
        F_reg = F + lambda * eye(trainsize1);

        inprods = h * (ps_train1' * ps_train2);
        cs = sum(F_reg \ inprods, 1)';
        normalizer = sum((1 - cs).^2);
        nu = (us_train2_tmpt - us_train1_tmpt * (F_reg \ inprods)) * (1 - cs);
        nu = nu / normalizer;
        
        G_pos_hat = (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ ps_train1');
        %pred_kernels{j, s} = G_pos_hat;
        
        % true Green's kernel
        modes = 100;
        G_pos_true = zeros(m, m);
        for k = 1:modes
            G_pos_true = G_pos_true + 2/L * sin(k*pi*x/L)' * sin(k*pi*x/L) * cos(k*pi*c*t(tmpt)/L);
        end
        
        % compute the relative error of the predicted Green's kernel
        kernel_errors(j, s) = norm(G_pos_hat - G_pos_true, 'fro') / norm(G_pos_true, 'fro');
        
        % learned operator is Psi_hat(g) = us_train * F_reg \ <f_i, g>
        % compute the mean relative train error
        inprods = h * (ps_train1' * ps_train);
        us_train_tmpt_hat = repmat(nu, [1, trainsize]) + (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
        l2norm =@(X) sqrt(sum(X.^2, 1));
        train_errors(j, s) = mean(l2norm(us_train_tmpt_hat - us_train_tmpt) ./ l2norm(us_train_tmpt));
        
        % compute the mean relative test error
        inprods = h * (ps_train1' * ps_test);
        us_test_tmpt_hat = repmat(nu, [1, testsize]) + (us_train1_tmpt - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
        test_errors(j, s) = mean(l2norm(us_test_tmpt_hat - us_test_tmpt) ./ l2norm(us_test_tmpt));
    end
end

ave_kernel_errors = mean(kernel_errors, 2);
ave_train_errors = mean(train_errors, 2);
ave_test_errors = mean(test_errors, 2);


figure(1)
plot(resolutions, ave_kernel_errors)
title('Green''s Kernel Error vs. Mesh Resolution', 'fontsize', 14)
xlabel('Resolution', 'fontsize', 14)
ylabel('Relative L^2 Kernel Error', 'fontsize', 14)

figure(2)
plot(resolutions, ave_train_errors)
title('Train Error vs. Mesh Resolution', 'fontsize', 14)
xlabel('Resolution', 'fontsize', 14)
ylabel('Relative L^2 Train Error', 'fontsize', 14)

figure(3)
plot(resolutions, ave_test_errors)
title('Test Error vs. Mesh Resolution', 'fontsize', 14)
xlabel('Resolution', 'fontsize', 14)
ylabel('Relative L^2 Test Error', 'fontsize', 14)

save('wave_BCs', 'kernel_errors', 'train_errors', 'test_errors', 'ps_trains', 'ps_tests', 'us_trains', 'us_tests', 'pred_nus', 'pred_kernels')
