clear all; close all; clc

m = 100;
L = 1;
h = L/(m-1);
x = linspace(0, L, m);

f = sin(x) + 3*cos(2*pi*x).^2;

c = 10;

sigma = 1;
K = h/(sqrt(2*pi)*sigma) * exp(-(x-x').^2/(2*sigma^2));

terms = 10;
M_invf = zeros(1, m);
%gkf = f;
ginvkf = f;
for k = 0:terms
    %M_invf = M_invf + (-1)^k * c^(-k-1) * gkf;
    %gkf = gkf * K;
    
    ginvkf = ginvkf * pinv(K);
    M_invf = M_invf + (-1)^k * c^(-k-1) * ginvkf;
end

plot(f)
hold on
plot(M_invf)
