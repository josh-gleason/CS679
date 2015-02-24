function run_part(params, samples, priors, g, b, axTitle)
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

    [~, n] = size(samples.s1);
    if priors.p1 == priors.p2
        NUM_SAMPLES1 = n;
        NUM_SAMPLES2 = n;
    elseif priors.p1 > priors.p2
        NUM_SAMPLES1 = n;
        NUM_SAMPLES2 = round(n * priors.p2 / priors.p1);
    else
        NUM_SAMPLES1 = round(n * priors.p1 / priors.p2);
        NUM_SAMPLES2 = n;
    end
    samples1 = samples.s1(:, 1:NUM_SAMPLES1);
    samples2 = samples.s2(:, 1:NUM_SAMPLES2);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % i) Design Bayes classifier for minimum error
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % See report

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % iii) Classify the samples by the classifier and count the number of
    %      misclassified samples
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % classify samples
    r1 = g(samples1, params, priors);
    r2 = g(samples2, params, priors);
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
    figure(); hold('on');
    % Plot the points
    plot(samples2(1,class2==1), samples2(2,class2==1), ' .', 'Color', [0.2 0.2 0.5]);
    plot(samples2(1,class2==2), samples2(2,class2==2), ' .', 'Color', [0.3 0.3 0.9]);
    plot(samples1(1,class1==2), samples1(2,class1==2), ' .', 'Color', [0.5 0.2 0.2]);
    plot(samples1(1,class1==1), samples1(2,class1==1), ' .', 'Color', [0.9 0.3 0.3]);
    axis('equal'); ax = axis();
    % Plot the decision boundry
    [x,y] = b(params, priors);
    plot(x, y, '-k', 'LineWidth', 2);
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
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % iv) Plot the Chernoff bound as a function of beta and find the
    %     optimum beta for the minimum P(error)
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    figure(); hold('on');
    % Compute error bound as a function of beta and find the Chernoff bound
    beta = 0:0.01:1;
    err = err_bound(beta, params, priors);
    [chernoff_bound, idx] = min(err);
    chernoff_beta = beta(idx);
    % Compute the Bhattacharyya bound
    bhattacharyya_beta = 0.5;
    bhattacharyya_bound = err_bound(bhattacharyya_beta, params, priors);
    % Approximate true P(error)
    err_truth = err_approx(params, priors);
    % Plot results
    plot(beta, err, '-k', 'LineWidth', 2);
    plot([bhattacharyya_beta([1,1]), 0], [0, bhattacharyya_bound([1,1])], '--k');
    plot([chernoff_beta([1,1]), 0], [0, chernoff_bound([1,1])], '--r');
    text(0.02, bhattacharyya_bound, 'Bhattacharyya', 'VerticalAlignment', 'bottom');
    text(0.02, chernoff_bound, 'Chernoff', 'VerticalAlignment', 'top', 'Color', 'Red');
    xlabel('\beta');
    ylabel('P(error) upper bound');
    title2 = sprintf(['%s\nBhattacharyya bound : P(error) \\leq %5.2f%%', ...
                      '\nChernoff bound : P(error) \\leq %5.2f%% (\\beta = %.2f)' ...
                      '\nTrue P(error) = %5.2f%%'], ...
                      axTitle, bhattacharyya_bound*100, chernoff_bound*100, chernoff_beta, err_truth*100);
    title(title2);
end
