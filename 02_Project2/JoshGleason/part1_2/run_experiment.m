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

%     NUM_EXP = 1000;
%     err_rate = zeros(4, NUM_EXP);
%     exp_range = 1:NUM_EXP;
%     for exp = exp_range

    % Number of samples per class
    NUM_SAMPLES = 10000;

    fprintf('-------------------------------------------------\n');
    fprintf('Problem %i\n', qnum);
    fprintf('-------------------------------------------------\n');
    
    % Load samples or generate if none exist
    fname1 = sprintf('samples_%i_1.txt', qnum);
    fname2 = sprintf('samples_%i_2.txt', qnum);
    if exist(fname1, 'file')
        samples.s1 = csvread(fname1);
    else
        samples.s1 = gen_samples(params.mu1, params.sigma1, NUM_SAMPLES);
        csvwrite(fname1, samples.s1);
    end

    if exist(fname2, 'file')
        samples.s2 = csvread(fname2);
    else
        samples.s2 = gen_samples(params.mu2, params.sigma2, NUM_SAMPLES);
        csvwrite(fname2, samples.s2);
    end

    for t_idx = 1:4
        n_training = NUM_SAMPLES / (10^(t_idx-1));
        
        title = sprintf('Problem %i: %i Training Samples', qnum, n_training);
        run_part(samples, n_training, title);
    end

end
