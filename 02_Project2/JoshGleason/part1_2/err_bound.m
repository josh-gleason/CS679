function err = err_bound(beta, params, priors)
%   Inputs
%       @beta    The value(s) of beta to evaluate the error bound for 2
%                normal distributions
%       @params  Struct containing the parameters of two 2D normal
%       |        distributions
%       +--params.mu1      The mean of distribution 1
%       +--params.sigma1   The covariance of distribution 1
%       +--params.mu2      The mean of distribution 2
%        +--params.sigma2   The covariance of distribution 2
%       @priors  Struct containing the priors to use for this experiment
%        +--priors.p1       The prior probability of state 1
%        +--priors.p2       The prior probability of state 2
%   Ouput
%       @err     Error bound
    mu1 = params.mu1;
    mu2 = params.mu2;
    s1 = params.sigma1;
    s2 = params.sigma2;
    p1 = priors.p1;
    p2 = priors.p2;
    n = numel(beta);
    % compute error
    err = zeros(1, n);
    for idx = 1:n
        b = beta(idx);
        k = (0.5*b*(1-b)*(mu1-mu2)'/((1-b)*s1+b*s2))*(mu1-mu2) + ...
             0.5*log(det((1-b)*s1 + b*s2)/(det(s1)^(1-b)*det(s2)^b));
        err(idx) = p1^b * p2^(1-b) * exp(-k);
    end
end
