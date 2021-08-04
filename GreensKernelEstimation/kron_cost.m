function cost = kron_cost(c, f, u, G1, G2, h, lambda)
   [~, trainsize] = size(c);
   cG1c = h^2 * c'*G1*c;
   fG2f = h^2 * f'*G2*f;
   
   cost = h/trainsize * sum(sum((u - h*G1*c*fG2f').^2));
   cost = cost + lambda*sum(sum(cG1c .* fG2f));
end