function [x, y] = boundry1(params, priors)
% BOUNDRY1 Get points to plot the boundry function for the given params as
%          a function and return the x, y values to plot it. Assumes Case 1
%          where all covariance matricies are equal to (sigma^2)*I
%   Inputs
%       @params Struct containing the parameters of two 2D normal
%       |       distributions
%       +--params.mu1      The mean of distribution 1
%       +--params.sigma1   The covariance of distribution 1
%       +--params.mu2      The mean of distribution 2
%       +--params.sigma2   The covariance of distribution 2
%       @priors  Struct containing the priors to use for this experiment
%       +--priors.p1       The prior probability of state 1
%       +--priors.p2       The prior probability of state 2
%   Outputs
%       @x      The points on the boundry in the x direction
%       @y      The points on the boundry in the y direction
    s = sqrt(params.sigma1(1,1));
    mu1 = params.mu1;
    mu2 = params.mu2;
    p1 = priors.p1;
    p2 = priors.p2;

    % decision boundry is a line
    w  = mu1 - mu2;
    x0 = (1 / 2) * (mu1 + mu2) - s^2 / (w'*w) * log(p1 / p2) * w;

    % decide how long the line should be (20 standard deviations plus gap)
    len = norm(w) + s * 20;

    % rotate direction of vector to be parallel with line
    wr = [0 -1; 1 0] * w / norm(w);
    
    % compute and return 2 points on the line
    a = x0 - wr*len/2;
    b = x0 + wr*len/2;
    x = [a(1) b(1)];
    y = [a(2) b(2)];
end
