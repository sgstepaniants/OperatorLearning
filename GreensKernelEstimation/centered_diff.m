function deriv = centered_diff(f, x, k)
    if k == 0
        deriv = f;
    else
        deriv = diff(f) ./ diff(x);
        deriv = [NaN, deriv] + [deriv, NaN];
        deriv(2:end-1) = deriv(2:end-1) / 2;
        deriv = [f; centered_diff(deriv, x, k-1)];
    end
end
