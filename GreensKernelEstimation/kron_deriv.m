function deriv = kron_deriv(c, f, u, G1, G2, h, lambda)
   [~, trainsize] = size(c);
   fG2f = h^2 * f'*G2*f;
   
   deriv = -2*h/trainsize*G1*u*fG2f';
   deriv = deriv + 2*h^2/trainsize*G1^2*c*fG2f'^2 + 2*h*lambda*G1*c*fG2f';
end