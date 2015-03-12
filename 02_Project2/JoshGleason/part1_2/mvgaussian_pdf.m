function y = mvgaussian_pdf(x, mu, sigma)
% MVGAUSSIAN_PDF Computes multivariate normal pdf
%   Inputs
%       @x      Samples in a NxM matrix where N is the dimensions and M is the
%               number of samples
%       @mu     The mean of the distribution
%       @sigma  The covariance matrix for the normal distribution
%   Outputs
%       @y      The density of the normal distribution at each column of x
    [n,~] = size(x);
    % subtract the mean
    x0 = bsxfun(@plus, -mu, x);
    % compute multivariate gaussian distribution
    y = 1/sqrt((2*pi)^n * det(sigma)) * exp(-1/2 * sum(x0.*(sigma\x0),1));
end
