function err_ratio = run_part(samples, n_training, axTitle)
% RUN_PART Run the experiments described in Project 1 part a.
%   @params  Struct containing the parameters of two 2D normal
%    |       distributions
%    +--params.mu1      The mean of distribution 1
%    +--params.sigma1   The covariance of distribution 1
%    +--params.mu2      The mean of distribution 2
%    +--params.sigma2   The covariance of distribution 2
%   @samples
%    +--s1              Set of samples from distribution 1
%    +--s2              Set of samples from distribution 2
%   @priors  Struct containing the priors to use for this experiment
%    +--priors.p1       The prior probability of state 1
%    +--priors.p2       The prior probability of state 2
%   @g       Pointer to the discriminant function to be used for
%            classification
%   @b       Pointer to the boundry plotting function
%   @axTitle Title to put on the plot

    NUM_SAMPLES1 = length(samples.s1);
    NUM_SAMPLES2 = length(samples.s2);

    training_samples1 = samples.s1(:, 1:n_training);
    training_samples2 = samples.s2(:, 1:n_training);

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % i) Estimate distribution using maximum likelihood estimation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Compute sample mean for both classes
    mu1 = sum(training_samples1, 2) / n_training;
    mu2 = sum(training_samples2, 2) / n_training;
    
    % Compute sample covariance for both classes
    s0_1 = bsxfun(@minus, training_samples1, mu1);
    s0_2 = bsxfun(@minus, training_samples2, mu2);
    
    covariance1 = zeros(2,2);
    covariance2 = zeros(2,2);
    for n = 1:n_training
        covariance1 = covariance1 + 1/(n_training-1) * s0_1(:,n)*s0_1(:,n)';
        covariance2 = covariance2 + 1/(n_training-1) * s0_2(:,n)*s0_2(:,n)';
    end
    
    % Display Results
    if exist('axTitle', 'var')
        fprintf('Class 1 mean = [ %8.5f, %8.5f ]\n', mu1(1), mu1(2));
        fprintf('Class 1 Covariance %5d training samples : [ %8.5f %8.5f ]\n', n_training, covariance1(1,1), covariance1(1,2));
        fprintf('                                            [ %8.5f %8.5f ]\n', covariance1(2,1), covariance1(2,2));
    
        fprintf('\n');

        fprintf('Class 2 mean = [ %8.5f, %8.5f ]\n', mu2(1), mu2(2));
        fprintf('Class 2 Covariance %5d training samples : [ %8.5f %8.5f ]\n', n_training, covariance2(1,1), covariance2(1,2));
        fprintf('                                            [ %8.5f %8.5f ]\n', covariance2(2,1), covariance2(2,2));

        fprintf('\n');
    end
    
    % Copy into params struct and use code from project 1
    params.mu1 = mu1;
    params.mu2 = mu2;
    params.sigma1 = covariance1;
    params.sigma2 = covariance2;
    params.p1 = 0.5;
    params.p2 = 0.5;
    
    samples1 = samples.s1;
    samples2 = samples.s2;
    
    priors.p1 = 0.5;
    priors.p2 = 0.5;
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % iii) Classify the samples by the classifier and count the number of
    %      misclassified samples
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % classify samples
    r1 = discriminant(samples1, params, priors);
    r2 = discriminant(samples2, params, priors);
    class1(r1 >  0) = 1;
    class1(r1 <= 0) = 2;
    class2(r2 >  0) = 1;
    class2(r2 <= 0) = 2;
    % count number of correct classifications
    correct1    = sum(class1==1);
    correct2    = sum(class2==2);
    % compute number of misclassified samples
    err_total   = (NUM_SAMPLES1 + NUM_SAMPLES2) - (correct1 + correct2);
    err_ratio   = err_total / (NUM_SAMPLES1 + NUM_SAMPLES2);
    err_percent = 100 * err_ratio;
        
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % ii) Plot the Bayes decision boundry together with the generated
    %     samples
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    if exist('axTitle', 'var')
        figure(); hold('on');
        % Plot the points
        plot(samples2(1,class2==1), samples2(2,class2==1), ' .', 'Color', [0.2 0.2 0.5]);
        plot(samples2(1,class2==2), samples2(2,class2==2), ' .', 'Color', [0.3 0.3 0.9]);
        plot(samples1(1,class1==2), samples1(2,class1==2), ' .', 'Color', [0.5 0.2 0.2]);
        plot(samples1(1,class1==1), samples1(2,class1==1), ' .', 'Color', [0.9 0.3 0.3]);
        axis('equal'); ax = axis();
        % Plot the decision boundry
        boundary_plot(params, priors); hold on;
        plot([params.mu1(1) params.mu2(1)], [params.mu1(2) params.mu2(2)], ' .k', 'Markersize', 25);
        axis(ax);
        h = legend('Misclassified \omega_{2} samples', ...
                   'Correct \omega_{2} samples', ...
                   'Misclassified \omega_{1} samples', ...
                   'Correct \omega_{1} samples', ...
                   'Decision Boundry', ...
                   'Mean of p(x|\omega_{i})', ...
                   'Location', 'EastOutside');
        % show results in title of figure
        title1 = sprintf('%s\nError rate: %5.2f%%', axTitle, err_percent);
        title(title1);
        % make room for legend in figure
        pos = get(gcf, 'Position'); pos(3) = pos(3)*1.5; set(gcf, 'Position', pos);
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % iv) Plot the Chernoff bound as a function of beta and find the
    %     optimum beta for the minimum P(error)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%     figure(); hold('on');
%     % Compute error bound as a function of beta and find the Chernoff bound
%     beta = 0:0.01:1;
%     err = err_bound(beta, params, priors);
%     [chernoff_bound, idx] = min(err);
%     chernoff_beta = beta(idx);
%     % Compute the Bhattacharyya bound
%     bhattacharyya_beta = 0.5;
%     bhattacharyya_bound = err_bound(bhattacharyya_beta, params, priors);
%     % Approximate true P(error)
%     err_truth = err_approx(params, priors);
%     % Plot results
%     plot(beta, err, '-k', 'LineWidth', 2);
%     plot([bhattacharyya_beta([1,1]), 0], [0, bhattacharyya_bound([1,1])], '--k');
%     plot([chernoff_beta([1,1]), 0], [0, chernoff_bound([1,1])], '--r');
%     text(0.02, bhattacharyya_bound, 'Bhattacharyya', 'VerticalAlignment', 'bottom');
%     text(0.02, chernoff_bound, 'Chernoff', 'VerticalAlignment', 'top', 'Color', 'Red');
%     xlabel('\beta');
%     ylabel('P(error) upper bound');
%     title2 = sprintf(['%s\nBhattacharyya bound : P(error) \\leq %5.2f%%', ...
%                       '\nChernoff bound : P(error) \\leq %5.2f%% (\\beta = %.2f)' ...
%                       '\nTrue P(error) = %5.2f%%'], ...
%                       axTitle, bhattacharyya_bound*100, chernoff_bound*100, chernoff_beta, err_truth*100);
%     title(title2);
end
