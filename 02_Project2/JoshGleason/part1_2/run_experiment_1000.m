function run_experiment(params, qnum)
% RUN_EXPERIMENT Run the project 1 experiments with the designated
%   parameters.
%   @params  Struct containing the parameters of two 2D normal
%    |       distributions
%    +--params.mu1      The mean of distribution 1
%    +--params.sigma1   The covariance of distribution 1
%    +--params.mu2      The mean of distribution 2
%    +--params.sigma2   The covariance of distribution 2
%   @g       Pointer to the discriminant function to be used for
%            classification
%   @b       Pointer to the boundry plotting function
%   @qnum    The question number to label the plot correctly

    NUM_EXP = 1000;
    err_rate = zeros(4, NUM_EXP);
    exp_range = 1:NUM_EXP;
    for exp = exp_range
        % Number of samples per class
        NUM_SAMPLES = 10000;

        samples.s1 = gen_samples(params.mu1, params.sigma1, NUM_SAMPLES);
        samples.s2 = gen_samples(params.mu2, params.sigma2, NUM_SAMPLES);

        for t_idx = 1:4
            n_training = NUM_SAMPLES / (10^(t_idx-1));      
            err_rate(t_idx, exp) = run_part(samples, n_training);
        end
    end
    
    errMax = max(err_rate(:));
    for t_idx = 1:4
        n_training = NUM_SAMPLES / (10^(t_idx-1));
        
        mean = sum(err_rate(t_idx,:)) / NUM_EXP;
        var = sum(err_rate(t_idx,:).^2) / (NUM_EXP-1) - mean^2;
        
        figure(); hold on;
        plot(exp_range, err_rate(t_idx, :), '-.k', 'LineWidth', 2);
        title(sprintf('Problem %i: %i Training Samples\nMean: %0.4f\nStdDev: %0.4f', qnum, n_training, mean, sqrt(var)));
        xlabel('Experiments');
        ylabel('Error Rate');
        axis([0 NUM_EXP 0 errMax]);
        
        figure(); hold on;
        hist(err_rate(t_idx, :), 50);
        title(sprintf('Problem %i: %i Training Samples', qnum, n_training));
        xlabel('Error Rate');
        ylabel('Experiments');
    end

end
