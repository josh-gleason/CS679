function bayesianTest2(truth_mu, truth_sigma, mu_mu, mu_sigma)
    if nargin == 0
        % defines our true p(x)
        truth_mu = 0;
        truth_sigma = 5;
        
        x_sigma = 5;
        
        % starting with mean estimated with error
        % define prior p(mu) = N(mu_mu, mu_sigma)
        mu_mu = -2;
        mu_sigma = 2;
    end

    % generate N random normally distributed samples
    N = 30;
    D = truth_mu + truth_sigma*randn(N,1);

    % compute some reasonable ranges to approximate x and mu over
    x_min = -5*truth_sigma + truth_mu;
    x_max = 5*truth_sigma + truth_mu;

    u_min = -5*mu_sigma + mu_mu;
    u_max = 5*mu_sigma + mu_mu;
    
    ZU = 500;
    ZX = 400;
    u = linspace(u_min, u_max, ZU);
    x = linspace(x_min, x_max, ZX);
    
    close all
    
    % compute p(x|D) iteratively (at each step)
    % initial estimate of mu is p(mu|D0) = p(mu)
    for n = 1:N
        %% Direct calculation of integral
        
        % Compute p(x,mu|D) = p(x|mu)*p(mu|D) = f(x,mu)
            % Compute p(x|mu) = f(x,mu) directly
            p_x_mu = zeros(ZX,ZU);
            for idx = 1:ZU
                p_x_mu(:,idx) = normpdf(x, u(idx), x_sigma);
            end
            
            % Compute p(mu|D) = p(D,mu)/p(D)
                % Compute p(D,mu) = p(D|mu)*p(mu)
                    % Compute p(D|mu)
                    p_D_mu = zeros(1,ZU);
                    for idx = 1:ZU
                        p_D_mu(idx) = prod(normpdf(D(1:n), u(idx), x_sigma));
                    end
                    
                    % Floating point errors may cause this to happen, this is where direct computation falls appart
                    if all(p_D_mu==0)
                        keyboard
                    end

                    % Compute p(mu)
                    p_mu = normpdf(u, mu_mu, mu_sigma);

                    % Put it together for p(D,mu)
                    p_Dmu = p_D_mu .* p_mu;

                % Compute p(D) = integral(p(D,mu), mu)
                p_D = trapz(u, p_Dmu);

                % Put it together for p(mu|D)
                p_mu_D = p_Dmu / p_D;

                % Shouldn't need to normalize for this to be true
                %assert(abs(trapz(u, p_mu_D)-1)<1e-10)
        
            % Put it together for p(x,mu|D)
            p_xmu_D = zeros(ZX,ZU);
            for idx = 1:ZX
                p_xmu_D(idx,:) = p_x_mu(idx,:) .* p_mu_D;
            end
            
        % Compute p(x|D) = integral(p(x,mu|D), mu)
        p_x_D = trapz(u, p_xmu_D, 2);

        %% Use the derived mu and sigma values
        % current estimated density function
        % p(mu|D) ~ N(mu_n, sigma_n)
        % Compute mean and standard deviation
        sample_mu = mean(D(1:n));

        mu_n = (n*mu_sigma^2)/(n*mu_sigma^2+x_sigma^2) * sample_mu + ...
                x_sigma^2 / (n*mu_sigma^2 + x_sigma^2) * mu_mu;
        sigma_n = sqrt(mu_sigma^2 * x_sigma^2 / (n*mu_sigma^2 + x_sigma^2));

        % Compute p(mu|D) directly
        p_mu_D_2 = normpdf(u, mu_n, sigma_n);
        
        % Compute p(x|D) directly
        p_x_D_2 = normpdf(x, mu_n, sqrt(sigma_n^2 + x_sigma^2));
        
        if n==1 || n==5 || n==10 || n==20 || n==N
            % Plot a posteriori for mean
            figure(1); hold on;
            plot(u, p_mu_D);
            figure(2); hold on;
            plot(u, p_mu_D_2);
            
            % Plot expected priors
            figure(3); hold on;
            plot(x, p_x_D);
            figure(4); hold on;
            plot(x, p_x_D_2);
        end
    end
    figure(1);
    grid on;
    title('Computed a-posteriori parameter estimate');
    xlabel('\mu');
    ylabel('P(\mu|\omega_i,D)');
    figure(2);
    grid on;
    title('Derived a-posteriori parameter estimate');
    xlabel('\mu');
    ylabel('P(\mu|\omega_i,D)');
    
    figure(3);
    grid on;
    title('Computed Prior');
    xlabel('x');
    ylabel('P(x|\omega_i,D)');
    % plot truth
    plot(x, normpdf(x, truth_mu, truth_sigma), '-r', 'LineWidth', 2);
    
    figure(4);
    grid on;
    title('Derived Prior');
    xlabel('x');
    ylabel('P(x|\omega_i,D)');
    % plot truth
    plot(x, normpdf(x, truth_mu, truth_sigma), '-r', 'LineWidth', 2);
end

function m = comp_mean(x, px)
    m = trapz(x,x.*px);
end

function y = normpdf(x, mu, sigma)
    y = 1/(sqrt(2*pi)*sigma) * exp(-1/2 * ((x-mu)/(sigma)).^2);
end