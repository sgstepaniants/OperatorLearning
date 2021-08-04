clear all; close all; clc

% Show the performance of the Green RKHS estimator over varying mesh sizes

% grid resolutions to test
resolutions = 50:50:1000;
numres = length(resolutions);

% Dirichlet boundary conditions
a = 1;
b = -1;

% regularizer
lambda = 0;

% number of modes to keep in KL expansion of Brownian bridge
modes = 100;

% number of trials to average over
numtrials = 100;

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
fs_trains = cell(numres, numtrials);
fs_tests = cell(numres, numtrials);
us_trains = cell(numres, numtrials);
us_tests = cell(numres, numtrials);
pred_nus = cell(numres, numtrials);
pred_kernels = cell(numres, numtrials);
for j=1:numres
    j
    for t=1:numtrials
        % number of grid points
        m = resolutions(j);
        L = 1;
        h = L/(m-1);
        x = linspace(0, L, m);
        
        % generate train and test input (forcing) functions as Brownian bridges
        ks = (1:modes) / L;
        fs_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, trainsize);
        fs_trains{j, t} = fs_train;
        fs_test = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*ks), pi*ks) * randn(modes, testsize);
        fs_tests{j, t} = fs_test;
        
        
        K = spdiags([-ones(m-2, 1), 2*ones(m-2, 1), -ones(m-2, 1)], -1:1, m-2, m-2);
        K = full(K) / h^2;
        
        % compute corresponding train and test solutions to Poisson equation
        us_train = zeros(m, trainsize);
        for k = 1:trainsize
            f = fs_train(2:m-1, k);
            f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
            us_train(:, k) = [a; K\f_pert; b];
        end
        us_trains{j, t} = us_train;
        
        us_test = zeros(m, testsize);
        for k = 1:testsize
            f = fs_test(2:m-1, k);
            f_pert = [f(1) + a/h^2; f(2:m-3); f(m-2) + b/h^2];
            us_test(:, k) = [a; K\f_pert; b];
        end
        us_tests{j, t} = us_test;
        
        % split train inputs and outputs into two groups
        fs_train1 = fs_train(:, traininds1);
        fs_train2 = fs_train(:, traininds2);
        us_train1 = us_train(:, traininds1);
        us_train2 = us_train(:, traininds2);
        
        % learn the Green's kernel of this Poisson BVP
        F = h * (fs_train1' * fs_train1);
        F_reg = F + lambda * eye(trainsize1);

        inprods = h * (fs_train1' * fs_train2);
        cs = sum(F_reg \ inprods, 1)';
        normalizer = sum((1 - cs).^2);
        nu = (us_train2 - us_train1 * (F_reg \ inprods)) * (1 - cs);
        nu = nu / normalizer;
        pred_nus{j, t} = nu;
        
        G_hat = (us_train1 - repmat(nu, [1, trainsize1])) * (F_reg \ fs_train1');
        pred_kernels{j, t} = G_hat;
        
        % true Green's kernel
        [X, Y] = meshgrid(x, x);
        G_true = (X + Y - abs(Y - X))/2 - X.*Y;
        
        % compute the relative error of the predicted Green's kernel
        kernel_errors(j, t) = norm(G_hat - G_true, 'fro') / norm(G_true, 'fro');
        
        % learned operator is Psi_hat(g) = us_train * F_reg \ <f_i, g>
        % compute the mean relative train error
        inprods = h * (fs_train1' * fs_train);
        us_train_hat = repmat(nu, [1, trainsize]) + (us_train1 - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
        l2norm =@(X) sqrt(sum(X.^2, 1));
        train_errors(j, t) = mean(l2norm(us_train_hat - us_train) ./ l2norm(us_train));
        
        % compute the mean relative test error
        inprods = h * (fs_train1' * fs_test);
        us_test_hat = repmat(nu, [1, testsize]) + (us_train1 - repmat(nu, [1, trainsize1])) * (F_reg \ inprods);
        test_errors(j, t) = mean(l2norm(us_test_hat - us_test) ./ l2norm(us_test));
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

save('poisson_BCs', 'kernel_errors', 'train_errors', 'test_errors', 'fs_trains', 'fs_tests', 'us_trains', 'us_tests', 'pred_nus', 'pred_kernels')
