function bayesianTest3(truth_mu, truth_sigma, mu_mu, mu_sigma, sigma_min, sigma_max)
    if nargin == 0
        % defines our true p(x)
        truth_mu = 0;
        truth_sigma = 3;
        
        % starting with mean estimated with error
        % define prior p(mu) = N(mu_mu, mu_sigma)
        mu_mu = 4;
        mu_sigma = 6;
        
        % prior knowledge on sigma follows uniform distribution
        % define p(sigma) = U(sigma_min, sigma_max)
        sigma_min = 1e-10; % must be greater than zero
        sigma_max = 10.0;
        
        % mu and sigma are independent
        % p(mu,sigma) = p(mu)*p(sigma)
    end
    
    % seed random number generator to ensure the same results each time
    rng(6675);

    % generate N random normally distributed samples
    N = 30;
    D = truth_mu + truth_sigma*randn(N,1);
    
    % compute some reasonable ranges to approximate x and mu over
    x_min = -3*sigma_max + mu_mu;
    x_max = 3*sigma_max + mu_mu;

    u_min = -3*mu_sigma + mu_mu;
    u_max = 3*mu_sigma + mu_mu;
    
    s_min = sigma_min;
    s_max = sigma_max;
    
    ZU = 200;
    ZS = 200;
    ZX = 200;
    u = linspace(u_min, u_max, ZU);
    s = linspace(s_min, s_max, ZS);
    x = linspace(x_min, x_max, ZX);
    
    [uu,ss] = meshgrid(u,s);
    uu = uu'; ss = ss';
    
    close all
    
    % compute p(x|D) iteratively (at each step)
    NV = [0 1 3 5 10 20 N];
    NVs = cellfun(@num2str, num2cell(NV), 'UniformOutput', false); % create cell array for legend
    for n = NV
        % Direct calculation of integral
        
        % Compute p(x,theta|D) = p(x|theta)*p(theta|D)  I.E. p(x,mu,sigma|D) = p(x|mu,sigma)*p(mu,sigma|D) = f(x,mu,sigma)
            % Compute p(x|theta) = p(x|mu,sigma) = f(x,mu,sigma) directly
            p_x_t = zeros(ZU,ZS,ZX);
            for idxU = 1:ZU
                for idxS = 1:ZS
                    p_x_t(idxU,idxS,:) = normpdf(x, u(idxU), s(idxS));
                end
            end
            
            % Compute p(theta|D) = p(D,theta)/p(D)  I.E. p(mu,sigma|D) = p(D,mu,sigma)/p(D)
            if n > 0
                % Compute p(D,theta) = p(D|theta)*p(theta)  I.E. p(D,mu,sigma) = p(D|mu,sigma)*p(mu,sigma)
                    % Compute p(D|theta)  I.E. p(D|mu,sigma)
                    p_D_t = zeros(ZU,ZS);
                    for idxU = 1:ZU
                        for idxS = 1:ZS
                            p_D_t(idxU, idxS) = prod(normpdf(D(1:n), u(idxU), abs(s(idxS))));
                        end
                    end
                    
                    % Floating point errors may cause this to happen, this is where direct computation falls appart
                    if all(p_D_t==0)
                        keyboard
                    end

                    % Compute p(theta)  I.E. p(mu,sigma)
                    p_t = normunifpdf(u, s, mu_mu, mu_sigma, sigma_min, sigma_max);

                    % Put it together for p(D,theta)  I.E. p(D,mu,sigma)
                    p_Dt = p_D_t .* p_t;

                % Compute p(D) = integral(p(D,theta), theta)  I.E. p(D) = integral(p(D,mu,sigma), mu, sigma)
                p_D = trapz(s, trapz(u, p_Dt));

                % Put it together for p(theta|D)  I.E. p(mu,sigma|D)
                p_t_D = p_Dt / p_D;
            else
                % No evidence so p(theta|D) = p(theta)
                p_t_D = normunifpdf(u, s, mu_mu, mu_sigma, sigma_min, sigma_max);
            end
            
            % Put it together for p(x,theta|D)  I.E. p(x,mu,sigma|D)
            p_xt_D = zeros(ZU,ZS,ZX);
            for idxX = 1:ZX
                p_xt_D(:,:,idxX) = p_x_t(:,:,idxX) .* p_t_D;
            end
            
        % Compute p(x|D) = integral(p(x,theta|D), mu)  I.E. p(x|D) = integral(p(x,mu,sigma|D), mu, sigma)
        p_x_D = reshape(trapz(s, trapz(u, p_xt_D)), 1, ZX);
        
        % Plot marginal distributions for sigma and mu
        figure(1); hold on;
        p_mu_D = trapz(s, p_t_D, 2);
        if n == 0 || n == 5 || n == N
            plot(u, p_mu_D, 'LineWidth', 2);
        else
            plot(u, p_mu_D);
        end

        figure(2); hold on;
        p_sigma_D = trapz(u, p_t_D, 1);
        if n == 0 || n == 5 || n == N
            plot(s, p_sigma_D, 'LineWidth', 2);
        else
            plot(s, p_sigma_D);
        end
        
        % Plot estimated prior distribution
        figure(3); hold on;
        if n == 0 || n == 5 || n == N
            plot(x, p_x_D, 'LineWidth', 2);
        else
            plot(x, p_x_D);
        end
    end
    
    % Compute mean and variance of last prior estimation
    mu_est = trapz(x, p_x_D.*x)
    sig_est = sqrt(var(x, p_x_D))
    
    figure(1);
    grid on;
    title('A-posteriori mean estimate');
    xlabel('\mu');
    ylabel('P(\mu|\omega_i,D)');
    % plot truth
    ax = axis;
    plot([truth_mu truth_mu], [0 ax(4)], '--r', 'LineWidth', 2); axis(ax);
    legend(NVs{:},'truth');
    
    figure(2);
    grid on;
    title('A-posteriori standard deviation estimate');
    xlabel('\sigma');
    ylabel('P(\sigma|\omega_i,D)');
    % plot truth
    ax = axis;
    plot([truth_sigma truth_sigma], [0 ax(4)], '--r', 'LineWidth', 2); axis(ax);
    legend(NVs{:},'truth');
    
    figure(3);
    grid on;
    title('Estimated Prior');
    xlabel('x');
    ylabel('P(x|\omega_i,D)');
    % plot truth
    plot(x, normpdf(x, truth_mu, truth_sigma), '--r', 'LineWidth', 2);
    legend(NVs{:},'truth');
end

function y = normpdf(x, mu, sigma)
    y = 1/(sqrt(2*pi)*sigma) * exp(-1/2 * ((x-mu)/(sigma)).^2);
end

function p = normunifpdf(x, y, mu, sigma, min_y, max_y)
% an independent normal with uniform joint density function
    p_x = normpdf(x, mu, sigma);    
    p_y = zeros(1, numel(y));
    p_y((y > min_y) & (y < max_y)) = 1 / (max_y - min_y);
    
    p = (p_x' * p_y);
    
    for idxX = 1:numel(x)
        for idxY = 1:numel(y)
            assert(p(idxX,idxY) == p_x(idxX)*p_y(idxY));
        end
    end
end
