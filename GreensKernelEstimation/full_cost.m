function cost = cost(c, fs_train_centered, us_train_centered, RK, h, lambda)
   [m, trainsize] = size(c);
   cf = sum(krondim(c, fs_train_centered), 2);
   cost = h/trainsize * sum(sum((us_train_centered - h^3 * reshape((RK*cf)', [m, m])' * fs_train_centered).^2));
   cost = cost + lambda*h^4*cf'*RK*cf;
end

%fun =@(c) h/trainsize * sum(sum((us_train_centered - h^3 * reshape((RK*sum(krondim(c, fs_train_centered), 2))', [m, m])' * fs_train_centered).^2)) + lambda*h^4*sum(krondim(c, fs_train_centered), 2)'*RK*sum(krondim(c, fs_train_centered), 2);