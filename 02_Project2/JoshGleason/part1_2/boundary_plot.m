function boundary_plot(params, priors)
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
    v1 = params.sigma1;
    v2 = params.sigma2;
    
    xmin = min(mux1(1) - 5*sqrt(v1(1,1)), mux2 - 5*sqrt(v2(1,1)));
    xmax = min(mux1(1) + 5*sqrt(v1(1,1)), mux2 + 5*sqrt(v2(1,1)));
    ymin = min(muy1(1) - 5*sqrt(v1(2,2)), muy2 - 5*sqrt(v2(2,2)));
    ymax = min(muy1(1) + 5*sqrt(v1(2,2)), muy2 + 5*sqrt(v2(2,2)));

    [x,y] = meshgrid( linspace(xmin, xmax, 1000), linspace(ymin, ymax, 1000) );
    
    v = [x(:)'; y(:)'];
    
    p1 = mvgaussian_pdf(v, mux1, v1);
    p2 = mvgaussian_pdf(v, mux2, v2);
    
    p1 = reshape(p1, [1000, 1000]);
    p2 = reshape(p2, [1000, 1000]);
    
    g1 = log(p1) + log(Pw1);
    g2 = log(p2) + log(Pw2);
    
    c = double(g1 > g2);
    contour(x, y, c, [0.5,0.5], 'LineWidth', 2, 'Color', [0,0,0]);
end
