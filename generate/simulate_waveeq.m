clear all; close all; clc

% load input functions
num_eigs = 100;
num_samples = 10000;
m = 20;
n = 20;
infilename = sprintf('KLE%d_n=%d_m=%dx%d.h5', num_eigs, num_samples, m, n);
infile = sprintf('./grfs/%s', infilename);

% input functions
inputs = h5read(infile, '/grfs');
fs = reshape(inputs', [n, m, num_samples]);

% mesh in space
xspan = h5read(infile, '/x')';

% mesh in time
tspan = h5read(infile, '/y')';
T = max(tspan);

% nonlinearity
alpha = 0;

sols = zeros(n, m, num_samples);
heatic =@(x) 0;
parfor k = 1:num_samples
    k
    currheatpde =@(x, t, u, dudx) heatpde(x, xspan, t, tspan, u, dudx, fs(:, :, k)', alpha);
    sol = pdepe(0, currheatpde, heatic, @heatbc, xspan, tspan);
    sols(:, :, k) = sol;
end

figure(1)
imagesc(fs(:, :, 1))
xlabel('x')
ylabel('t')

figure(2)
imagesc(sols(:, :, 1))
xlabel('x')
ylabel('t')

% save solutions
outputs = reshape(sols, [], num_samples)';

outfile = sprintf('heateq_%s', infilename);
h5create(outfile, '/input', size(inputs))
h5write(outfile, '/input', inputs)

h5create(outfile, '/output', size(outputs))
h5write(outfile, '/output', outputs)

h5create(outfile, '/x', size(xspan))
h5write(outfile, '/x', xspan)

h5create(outfile, '/t', size(tspan))
h5write(outfile, '/t', tspan)

h5create(outfile, '/alpha', 1)
h5write(outfile, '/alpha', alpha)


function [c, f, s] = heatpde(x, xspan, t, tspan, u, dudx, forcing, alpha)
    c = 1;
    f = exp(alpha*u).*dudx;
    
    [~, x_ind] = min(abs(xspan - x));
    [~, t_ind] = min(abs(tspan - t));
    s = forcing(x_ind(1), t_ind(1));
end

function [pl, ql, pr, qr] = heatbc(xl, ul, xr, ur, t)
    pl = ul;
    ql = 0;
    pr = ur;
    qr = 0;
end