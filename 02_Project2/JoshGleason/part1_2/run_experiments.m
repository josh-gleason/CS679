function run_experiments(params, qnum)
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

    % Number of samples per class
    NUM_SAMPLES = 10000;

    SAMPLES = [3 4 5 6 7 8 9 10 15 20 40 60 100 200 500 1000 5000 10000];
    N_SAMPLE_TRIALS = numel(SAMPLES);
    
    NUM_EXP = 1000;
    err_rate = zeros(N_SAMPLE_TRIALS, NUM_EXP);
    exp_range = 1:NUM_EXP;
    start_t = tic;
    for exp = exp_range
        samples.s1 = gen_samples(params.mu1, params.sigma1, NUM_SAMPLES);
        samples.s2 = gen_samples(params.mu2, params.sigma2, NUM_SAMPLES);

        for t_idx = 1:N_SAMPLE_TRIALS
            n_training = SAMPLES(t_idx);
            err_rate(t_idx, exp) = run_part(samples, n_training);
        end
        
        fprintf('Time so far %0.8fs\n', toc(start_t));
        fprintf('Completed %0.4f%%\n', 100*exp/NUM_EXP);
    end
    
%    errMax = max(err_rate(:));
    mean = sum(err_rate(:,:),2) / NUM_EXP;
    var = sum(err_rate(:,:).^2,2) / (NUM_EXP-1) - mean.^2;
%     for t_idx = 1:N_SAMPLE_TRIALS
%         n_training = SAMPLES(t_idx);
%         
%         figure(); hold on;
%         plot(exp_range, err_rate(t_idx, :), '-.k', 'LineWidth', 2);
%         title(sprintf('Problem %i: %i Training Samples\nMean: %0.4f\nStdDev: %0.4f', qnum, n_training, mean(t_idx), sqrt(var(t_idx))));
%         xlabel('Experiments');
%         ylabel('Error Rate');
%         axis([0 NUM_EXP 0 errMax]);
%         
%         figure(); hold on;
%         hist(err_rate(t_idx, :), 50);
%         title(sprintf('Problem %i: %i Training Samples', qnum, n_training));
%         xlabel('Error Rate');
%         ylabel('Experiments');
%     end
    
    figure();
    semilogx(SAMPLES, mean, 'k', 'LineWidth', 2); hold on;
    semilogx(SAMPLES, mean + sqrt(var), 'b');
    semilogx(SAMPLES, mean - sqrt(var), 'b');
    ax = axis();
    axis([SAMPLES(1) SAMPLES(end) ax(3) ax(4)]);
    fill([SAMPLES fliplr(SAMPLES)],  [(mean+sqrt(var))' fliplr((mean-sqrt(var))')], [0.2,0.2,0.8]);
    alpha(0.4);
    title(sprintf('Problem %i: Effect of training data size on Error Rate', qnum));
    xlabel('Number of Training Samples');
    ylabel('Classifier Error Rate');
    legend('Mean Error Rate', '1 Standard Deviation from mean');
    
    figure();
    loglog(SAMPLES, mean, 'k', 'LineWidth', 2); hold on;
    loglog(SAMPLES, mean + sqrt(var), 'b');
    loglog(SAMPLES, mean - sqrt(var), 'b');
    ax = axis();
    axis([SAMPLES(1) SAMPLES(end) ax(3) ax(4)]);
    fill([SAMPLES fliplr(SAMPLES)],  [(mean+sqrt(var))' fliplr((mean-sqrt(var))')], [0.2,0.2,0.8]);
    alpha(0.4);
    title(sprintf('Problem %i: Effect of training data size on Error Rate', qnum));
    xlabel('Number of Training Samples');
    ylabel('Classifier Error Rate');
    legend('Mean Error Rate', '1 Standard Deviation from mean');
end
