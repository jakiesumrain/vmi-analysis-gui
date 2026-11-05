function Pn = leg(n, x)
    % Compute the Legendre polynomial of a specific order n at points x
    % Inputs:
    %   n - order of the Legendre polynomial (non-negative integer)
    %   x - vector of points at which to evaluate the polynomial
    % Output:
    %   Pn - values of the Legendre polynomial P_n(x) at the points x

    % Handle base cases
    if n == 0
        Pn = ones(size(x));  % P_0(x) = 1
        return;
    elseif n == 1
        Pn = x;              % P_1(x) = x
        return;
    end

    % Initialize the first two polynomials
    P_prev = ones(size(x));  % P_0(x) = 1
    P_curr = x;              % P_1(x) = x

    % Use the recurrence relation to compute P_n(x)
    for k = 2:n
        P_next = ((2*k-1) * x .* P_curr - (k-1) * P_prev) / k;
        P_prev = P_curr;
        P_curr = P_next;
    end

    Pn = P_curr;  % P_n(x)
end