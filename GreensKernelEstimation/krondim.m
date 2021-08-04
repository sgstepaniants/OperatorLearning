function prod = krondim(u, v)
    [n, m] = size(u);
    prod = zeros(n^2, m);
    for k = 1:m
        prod(:, k) = kron(u(:, k), v(:, k));
    end
end
