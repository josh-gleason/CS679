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

    NUM_STEPS = 1000;

    mux1 = params.mu1(1);
    muy1 = params.mu1(2);
    mux2 = params.mu2(1);
    muy2 = params.mu2(2);  
    v1 = params.sigma1;
    v2 = params.sigma2;
    
    xmin = min(mux1(1) - 15*sqrt(v1(1,1)), mux2 - 15*sqrt(v2(1,1)));
    xmax = min(mux1(1) + 15*sqrt(v1(1,1)), mux2 + 15*sqrt(v2(1,1)));
    ymin = min(muy1(1) - 15*sqrt(v1(2,2)), muy2 - 15*sqrt(v2(2,2)));
    ymax = min(muy1(1) + 15*sqrt(v1(2,2)), muy2 + 15*sqrt(v2(2,2)));

    [x,y] = meshgrid( linspace(xmin, xmax, NUM_STEPS), linspace(ymin, ymax, NUM_STEPS) );
    
    v = [x(:)'; y(:)'];
    
    g = reshape(discriminant(v, params, priors), [NUM_STEPS, NUM_STEPS]);

    contour(x, y, g, [0,0], 'LineWidth', 2, 'Color', [0,0,0], 'Clipping', 'off');
end
