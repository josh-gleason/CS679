% Generate n samples from a multivariate normal distribution using the
% provided mean (mu) and covariance matrix (sigma).
function samples = gen_samples(mu, sigma, n)
    dim = numel(mu);
    % Sample the unit normal distribution
    samples = sample_unit_normal([dim, n]);
    % Color noise using inverse whitening transform
    [V,D] = eig(sigma);
    Ainv = sqrt(D)*V';
    for idx=1:n
        samples(:,idx) = (Ainv' * samples(:,idx)) + mu;
    end
end

% Generate samples from the unit normal distribution and store into an
% array of arbitrary size
function samples = sample_unit_normal(shape)
    n = prod(shape);
    % Generate n samples using box_muller transform
    s = sample_box_muller(n);
    % Reshape into n x dim array
    samples = reshape(s, shape);
end

% Generate n samples from the 1d unit normal distribution using the 
% box muller transform
function s = sample_box_muller(n)
    % generate 2*n random variables from uniform(0,1) distribution
    u = rand([2,n]);

    % If the unlikely case occurs where a value is actually zero
    % then make choose another random number
    idx = find(u(1,:)==0);
    while ~isempty(idx)
        u(1,idx) = rand([1,numel(idx)]);
        idx = find(u(1,:)==0);
    end

    % Box-muller transform
    u(1,:) = -2 * log(u(1,:));
    u(2,:) = 2*pi*u(2,:);
    s = sqrt(u(1,:)) .* cos(u(2,:));
end
