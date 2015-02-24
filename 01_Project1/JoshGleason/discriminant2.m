function r = discriminant2(x, params, priors)
%DISCRIMINANT2 Compute the difference between discriminant functions
%              assuming two classes and arbitrary covariance matricies
%    Inputs
%       @x      Samples in a dxM matrix where d is the dimensions and M is
%               the number of samples
%       @params Struct containing the parameters of two 2D normal
%       |       distributions
%       +--params.mu1      The mean of distribution 1
%       +--params.sigma1   The covariance of distribution 1
%       +--params.mu2      The mean of distribution 2
%       +--params.sigma2   The covariance of distribution 2
%       @priors  Struct containing the priors to use for this experiment
%       +--priors.p1       The prior probability of state 1
%       +--priors.p2       The prior probability of state 2
%   Output
%       @r      The difference between the discriminant functions of
%               distribution 1 and 2: r = g1(x) - g2(x). Choose category 1
%               for r > 0 and category 2 otherwise.
    g1 = g(x, params.mu1, params.sigma1, priors.p1);
    g2 = g(x, params.mu2, params.sigma2, priors.p2);
    r = g1 - g2;
end

function p = g(x, mu, s, prior)
    % slope and bias
    W  = -0.5*inv(s);
    w  = s\mu;
    w0 = -0.5*(mu'/s)*mu - 0.5*log(det(s)) + log(prior);
    % compute discriminator value for each x
    [~,n] = size(x);
    p = zeros(1, n);
    for idx=1:n
        p(idx) = (x(:,idx)'*W)*x(:,idx) + w'*x(:,idx) + w0;
    end
end
