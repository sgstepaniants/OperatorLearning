function U = wave_eq(p, v, c, x, t)
    % Input  -- f = u(x,0)
    %        -- g = ut(x,0)
    %        -- c = the speed constant in wave equation
    %        -- L = length of domain
    %        -- T = length of time interval
    % Output -- U solution matrix
    
    m = length(x);
    n = length(t);
    
    h = x(2) - x(1);
    dt = t(2) - t(1);
    
    % Initialize parameters and U
    r = c * dt/h;
    U = zeros(m, n);
    
    % Compute first and second rows
    U(2:m-1, 1) = p(2:m-1);
    U(2:m-1, 2) = (1-r^2)*p(2:m-1) + dt*v(2:m-1) + r^2/2*(p(3:m) + p(1:m-2));
    
    % Compute remaining rows of U 
    for j = 3:n
        for i = 2:(m-1)
            U(i,j) = (2 - 2*r^2) * U(i,j-1) + r^2 * (U(i-1,j-1) + U(i+1,j-1)) - U(i,j-2);
        end
    end
