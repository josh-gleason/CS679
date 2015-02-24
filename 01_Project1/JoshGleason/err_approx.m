function err = err_approx(params, priors)
%ERR_APPROX Approximate the P(error) by integrating P(error|x) across a
%           significant area.
%   Inputs
%       @params  Struct containing the parameters of two 2D normal
%       |       distributions
%       +--params.mu1      The mean of distribution 1
%       +--params.sigma1   The covariance of distribution 1
%       +--params.mu2      The mean of distribution 2
%       +--params.sigma2   The covariance of distribution 2
%       @priors  Struct containing the priors to use for this experiment
%       +--priors.p1       The prior probability of state 1
%       +--priors.p2       The prior probability of state 2
%       @g      Pointer to discriminat function
%   Outputs
%       @err    Total expected error-rate
    sx1 = sqrt(params.sigma1(1,1));
    sy1 = sqrt(params.sigma1(2,2));
    sx2 = sqrt(params.sigma2(1,1));
    sy2 = sqrt(params.sigma2(2,2));
    mux1 = params.mu1(1);
    muy1 = params.mu1(1);
    mux2 = params.mu2(2);
    muy2 = params.mu2(2);
    
    xmin = min(mux1 - sx1*5, mux2 - sx2*5);
    xmax = max(mux1 + sx1*5, mux2 + sx2*5);
    ymin = min(muy1 - sy1*5, muy2 - sy2*5);
    ymax = max(muy1 + sy1*5, muy2 + sy2*5);
    
    NUM_PTS = 500;
    
    x = linspace(xmin, xmax, NUM_PTS);
    y = linspace(ymin, ymax, NUM_PTS);
    [xx,yy] = meshgrid(x, y);
    pts = [reshape(xx, [1,NUM_PTS^2]); reshape(yy, [1,NUM_PTS^2])];
    % p(x|w1) and p(x|w2)
    px1 = mvgaussian_pdf(pts, params.mu1, params.sigma1);
    px2 = mvgaussian_pdf(pts, params.mu2, params.sigma2);
    % P(error|x)p(x) (using minimum error-rate rules)
    pErrorXpx = min(px1*priors.p1, px2*priors.p2);
    pErrorXpx = reshape(pErrorXpx, [NUM_PTS, NUM_PTS]);
    % Integrate over P(error|x)
    err = trapz(x, trapz(y, pErrorXpx));
end