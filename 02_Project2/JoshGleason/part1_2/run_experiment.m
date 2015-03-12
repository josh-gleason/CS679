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

    % Number of samples per class
    NUM_SAMPLES = 10000;
    
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

    titleA = sprintf('Problem %ia', qnum);
    titleB = sprintf('Problem %ib', qnum);

    n_training1 = NUM_SAMPLES;
    n_training2 = round(NUM_SAMPLES / 1000);

    % estimate(training_samples, test_samples)
    run_part(samples, n_training1, titleA);
    run_part(samples, n_training2, titleB);
end
