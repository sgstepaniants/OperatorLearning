function gf = gaussConv(f, x, sigma)
    h = x(2) - x(1);
    gf = f * h/(sqrt(2*pi)*sigma) * exp(-(x-x').^2/(2*sigma^2));
end
