function C = crosscov(X, Y)
    C = bsxfun(@minus, X, mean(X, 1))' * bsxfun(@minus, Y, mean(Y, 1)) / size(X, 1);
end
