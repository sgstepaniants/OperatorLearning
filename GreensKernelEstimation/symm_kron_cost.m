function cost = symm_kron_cost(c, f, u, G1, G2, h, lambda)
   [~, trainsize] = size(c);
   cG1c = h^2 * c'*G1*c;
   fG1c = h^2 * f'*G1*c;
   fG2f = h^2 * f'*G2*f;
   fG2c = h^2 * f'*G2*c;
   cG2f = h^2 * c'*G2*f;
   
   cost = h/trainsize * sum(sum((u - h/2*(G1*c*fG2f' + G1*f*fG2c')).^2));
   cost = cost + lambda/2*sum(sum(cG1c.*fG2f + fG1c.*cG2f));
end