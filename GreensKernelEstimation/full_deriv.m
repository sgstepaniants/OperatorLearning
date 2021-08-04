function deriv = deriv(c, fs_train_centered, us_train_centered, RK, h, lambda)
   [m, trainsize] = size(c);
   uf = sum(krondim(us_train_centered, fs_train_centered), 2);
   cf = sum(krondim(c, fs_train_centered), 2);
   RKfcf = reshape((RK*cf)', [m, m])' * fs_train_centered;
   fRKfcf = sum(krondim(RKfcf, fs_train_centered), 2);
   
   deriv = -2*h^3/trainsize * reshape((RK*uf)', [m, m])' * fs_train_centered;
   deriv = deriv + 2*h^6/trainsize * reshape((RK*fRKfcf)', [m, m])' * fs_train_centered;
   deriv = deriv + 2*lambda*h^3 * reshape((RK*cf)', [m, m])' * fs_train_centered;
end

%uf = sum(krondim(us_train_centered, fs_train_centered), 2);
%grad =@(c) -2*h^3/trainsize * reshape((RK*uf)', [m, m])' * fs_train_centered + 2*h^6/trainsize * reshape((RK*sum(krondim(reshape((RK*sum(krondim(c, fs_train_centered), 2))', [m, m])' * fs_train_centered, fs_train_centered), 2))', [m, m])' * fs_train_centered + 2*lambda*h^3 * reshape((RK*sum(krondim(c, fs_train_centered), 2))', [m, m])' * fs_train_centered;