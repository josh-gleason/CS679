function [x, y] = boundry2(params, priors)
% BOUNDRY2 Get points to plot the boundry function for the given params as
%          a function and return the x, y values to plot it. This will
%          work given the following constraints
%       params.sigma1 is diagonal 2x2 covariance matrix
%       params.sigma2 is a diagonal 2x2 covariance matrix ~= sigma1
%       The decision boundry must be an ellipse or this function will halt
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
%       @x      The x components of the decision boundry
%       @y      The y components of the decision boundry

    mux1 = params.mu1(1);
    muy1 = params.mu1(2);
    mux2 = params.mu2(1);
    muy2 = params.mu2(2);
    Pw1 = priors.p1;
    Pw2 = priors.p2;   
    sx1 = sqrt(params.sigma1(1,1));
    sy1 = sqrt(params.sigma1(1,1));
    sx2 = sqrt(params.sigma2(1,1));
    sy2 = sqrt(params.sigma2(2,2));
    
    % solve for center and radi of ellipse (symbolically solved prior)
    cx = (mux2*sx1^2 - mux1*sx2^2)/(sx1^2 - sx2^2);
    cy = (muy2*sy1^2 - muy1*sy2^2)/(sy1^2 - sy2^2);
    rx2 = (log((Pw1*sx2*sy2)/(Pw2*sx1*sy1)) + (mux1/sx1^2 - mux2/sx2^2)^2/(2/sx1^2 - 2/sx2^2) + (muy1/sy1^2 - muy2/sy2^2)^2/(2/sy1^2 - 2/sy2^2) - mux1^2/(2*sx1^2) + mux2^2/(2*sx2^2) - muy1^2/(2*sy1^2) + muy2^2/(2*sy2^2))/(1/(2*sx1^2) - 1/(2*sx2^2));
    ry2 = (log((Pw1*sx2*sy2)/(Pw2*sx1*sy1)) + (mux1/sx1^2 - mux2/sx2^2)^2/(2/sx1^2 - 2/sx2^2) + (muy1/sy1^2 - muy2/sy2^2)^2/(2/sy1^2 - 2/sy2^2) - mux1^2/(2*sx1^2) + mux2^2/(2*sx2^2) - muy1^2/(2*sy1^2) + muy2^2/(2*sy2^2))/(1/(2*sy1^2) - 1/(2*sy2^2));

    % Verify this is an ellipse
    assert(rx2 > 0 && ry2 > 0);
    
    rx = sqrt(rx2);
    ry = sqrt(ry2);

    % parametric equation for an ellipse
    t = -pi:0.01:pi;
    x = cx+rx*cos(t);
    y = cy+ry*sin(t);
end
