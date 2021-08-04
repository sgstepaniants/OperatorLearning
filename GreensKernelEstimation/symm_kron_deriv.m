function deriv = symm_kron_deriv(c, f, u, G1, G2, h, lambda)
   [~, trainsize] = size(c);
   fG2f = h^2 * f'*G2*f;
   fG2u = h^2 * f'*G2*u;
   fG2c = h^2 * f'*G2*c;
   
   deriv = -h/trainsize * (G1*u*fG2f' + G1*f*fG2u');
   deriv = deriv + 1/(2*trainsize) * (h^2*G1^2*c*fG2f'^2 + h^4*G1*f*(f'*G2*G1*c*fG2f')' + h^2*G1^2*f*fG2c'*fG2f' + h^4*G1*f*(f'*G2*G1*f*fG2c')');
   deriv = deriv + h*lambda * (G1*c*fG2f' + G1*f*fG2c');
end