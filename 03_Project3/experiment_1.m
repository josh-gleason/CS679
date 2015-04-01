function experiment_1(args)
%EXPERIMENT_1 Run experiments a.II-a.IV
    
    % Load training data
    training = load_training(args);
    
    % Load test images
    [test_samples, test_labels, img_size] = load_dataset(args.testdir);
    assert(all(img_size == training.img_size));
    
    % Subtract mean from test images
    test_samples = bsxfun(@minus, test_samples, training.mean);
    
    % Choose the top eigenvectors preserving some set amount of information
    k = find(training.information >= args.information, 1, 'first');
    U = training.eigenvectors(:,1:k);

    % Compute the covariance 
    covariance = U'*U*diag(training.eigenvalues(1:k));
    
    % Project both training and testing to eigenspace
    train_b = U.' * training.samples;
    test_b = U.' * test_samples;
    
    [~,n_train_imgs] = size(train_b);
    [~,n_test_imgs] = size(test_b);
    
    % calculate Mahalanobis distance between each training and test image
    d = zeros(n_train_imgs, n_test_imgs);
    for idx = 1:n_train_imgs
        % subtract train_b(:,idx) from each test_b
        db = bsxfun(@minus, test_b, train_b(:,idx));
        d(idx,:) = sqrt(sum(db .* (covariance \ db), 1));
    end
    
    % Sort each column by nearest match
    [~, d_idx] = sort(d);
    
    % Convert to matrix labels that were matched to
    matched_labels = training.labels(d_idx);
    
    % This is the correct labels matrix
    correct_labels = repmat(test_labels, length(training.labels), 1);
    
    % Find correct matches in the matched labels matrix
    correct = matched_labels == correct_labels;
    
    % This fills in 1s below the correct matches
    correct = cumsum(correct);
    correct(correct > 1) = 1;
    
    % Compute CMC
    cmc = sum(correct,2) / n_test_imgs;
    
    % Plot CMC results
    h = figure();
    plot(1:50, cmc(1:50), '-', 'Marker', '.', 'LineWidth', 2, 'MarkerSize', 20);
    title('Comparitive CMC graph');
    xlabel('Rank');
    ylabel('Performace');
    legend(sprintf('Information %0.2f', training.information(k)), ...
        'Location', 'southeast');
    ax = axis();
    axis([ax(1) ax(2) ax(3) 1]); % always go up to 1
    % Save figure
    savefig(h, [args.resultsdir filesep 'CMC.fig']);
    print(h, [args.resultsdir filesep 'CMC.png'], '-dpng');
    
    % Save CMC to plot later
    save([args.resultsdir filesep 'CMC.mat'], 'cmc');
    
    % Get 3 correctly and incorrectly matched images
    correct_idx = find(correct(1,:), 3, 'first');
    correct_match_idx = matched_labels(1,correct_idx);

    correct1 = get_test_image(test_samples, training, correct_idx(1));
    correct2 = get_test_image(test_samples, training, correct_idx(2));
    correct3 = get_test_image(test_samples, training, correct_idx(3));

    correct_match_1 = get_training_image(training, correct_match_idx(1));
    correct_match_2 = get_training_image(training, correct_match_idx(2));
    correct_match_3 = get_training_image(training, correct_match_idx(3));

    h = figure();
    subplot(2,3,1);
    imshow(correct1);
    title('Correct 1');
    subplot(2,3,2);
    imshow(correct2);
    title('Correct 2');
    subplot(2,3,3);
    imshow(correct3);
    title('Correct 3');
    subplot(2,3,4);
    imshow(correct_match_1);
    title('Match 1');
    subplot(2,3,5);
    imshow(correct_match_2);
    title('Match 2');
    subplot(2,3,6);
    imshow(correct_match_3);
    title('Match 3');
    % Save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) p(3) 285]);
    savefig(h, [args.resultsdir filesep 'CorrectMatches.fig']);
    print(h, [args.resultsdir filesep 'CorrectMatches.png'], '-dpng')
    
    % Get 3 incorrectly matched images
    incorrect_idx = find(~correct(1,:), 3, 'first');
    incorrect_match_idx = matched_labels(1,incorrect_idx);
    
    incorrect1 = get_test_image(test_samples, training, incorrect_idx(1));
    incorrect2 = get_test_image(test_samples, training, incorrect_idx(2));
    incorrect3 = get_test_image(test_samples, training, incorrect_idx(3));

    incorrect_match_1 = get_training_image(training, incorrect_match_idx(1));
    incorrect_match_2 = get_training_image(training, incorrect_match_idx(2));
    incorrect_match_3 = get_training_image(training, incorrect_match_idx(3));
    
    h = figure();
    subplot(2,3,1);
    imshow(incorrect1);
    title('Incorrect 1');
    subplot(2,3,2);
    imshow(incorrect2);
    title('Incorrect 2');
    subplot(2,3,3);
    imshow(incorrect3);
    title('Incorrect 3');
    subplot(2,3,4);
    imshow(incorrect_match_1);
    title('Match 1');
    subplot(2,3,5);
    imshow(incorrect_match_2);
    title('Match 2');
    subplot(2,3,6);
    imshow(incorrect_match_3);
    title('Match 3');
    % Save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) p(3) 285]);
    savefig(h, [args.resultsdir filesep 'IncorrectMatches.fig']);
    print(h, [args.resultsdir filesep 'IncorrectMatches.png'], '-dpng')
end
