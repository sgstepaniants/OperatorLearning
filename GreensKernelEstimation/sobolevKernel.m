clear all; close all; clc

a = 0;
b = 1;

m = 3;

s = 0:m-1;
k = 0:m-1;
[S, K] = meshgrid(s, k);

L = (-1).^(K-S).*(b-a).^(K-S)./factorial(max(K-S, 0));
L = tril(L);

U = (b-a).^(S-K)./factorial(max(S-K, 0));
U = triu(U);

M = (-1).^(m-S-1).*(b-a).^(2*m-S-K-1)./factorial(2*m-S-K-1);


n = 10000;
x = linspace(a, b, n);

% solve for the left-hand side values of the reproducing kernel (in 1D)
krep = repmat(k', 1, n);
xrep = repmat(x, m, 1);
vx = (-1).^krep.*max(b-x, 0).^krep./factorial(krep) - (-1)^m.*max(b-x, 0).^(2*m-krep-1)./factorial(2*m-krep-1);
Kxa = (L + U + M) \ vx;

%imagesc(Kxa)

% plot(Kxa)
% hold on
% plot((1+b-x)/(2+b-a))

% reconstruct the reproducing kernel
[X, Y] = meshgrid(x, x);

krep = repmat(k, n, 1);
yrep = repmat(x', 1, m);
A = (yrep-a).^krep./factorial(krep) + (-1).^(m-krep-1).*(yrep-a).^(2*m-krep-1)./factorial(2*m-krep-1);
K = (-1)^m/factorial(2*m-1)*max(Y-X, 0).^(2*m-1) + A*Kxa;

%K_true = -max(Y-X, 0) + (1+b-X)/(2+b-a).*(1+Y-a);

% plot the reproducing kernel
figure(1)
imagesc(K)
colorbar()

% generate random test functions
samples = 100;
modes = 100;
freqs = (1:modes) / (b-a);
fs_train = sqrt(2)*bsxfun(@rdivide, sin(pi*x'*freqs), pi*freqs) * randn(modes, samples);

f = fs_train(:, 1)';
%f = sin(3*pi*x);
f_derivs = centered_diff(f, x, m);
f_derivs(isnan(f_derivs)) = 0;
f_pred = zeros(1, n);
for j = 1:n
    Kxj_derivs = centered_diff(K(j, :), x, m);
    Kxj_derivs(isnan(Kxj_derivs)) = 0;
    
    ind_a = sub2ind([m+1, n], 1:m, 1:m);
    ind_b = sub2ind([m+1, n], 1:m, n:-1:n-m+1);
    inprod_xj = sum(f_derivs([ind_a, ind_b]) .* Kxj_derivs([ind_a, ind_b]));
    inprod_xj = inprod_xj + sum(f_derivs(m+1, :) .* Kxj_derivs(m+1, :)) * (b-a)/n;
    f_pred(j) = inprod_xj;
end

figure(2)
plot(f)
hold on
plot(f_pred)


%A = (b - a).^(K+m-S) ./ factorial(max(K+m-S, 0));
%A = tril(A);


N = 10;
v = 0:(N-1);
[X, Y] = meshgrid(v, v);

S1 = sin(pi/(N+1).*(X+1).*(Y+1)); % DST-I normalization 2/(N+1)
S2 = sin(pi/N.*(X+1/2).*(Y+1)); % DST-II normalization 2/N
S3 = sin(pi/N.*(X+1).*(Y+1/2));
S3(:, end) = S3(:, end) / 2; % DST-III normalization 2/N
S4 = sin(pi/N.*(X+1/2).*(Y+1/2)); % DST-I normalization 2/N
