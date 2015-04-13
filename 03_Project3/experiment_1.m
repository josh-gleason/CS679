function experiment_1(args)
%EXPERIMENT_1 Run experiments a.II-a.IV
    
    % Load training data
    training = load_training(args);
    
    % Load test images
    test_data = load_test_data(args.testdir, training.mean);
    
    % Compare each test sample with each training sample in eigenspace
    [dist, U, train_b, test_b] = compare_samples(training, test_data, args.information);

    n_test_imgs = length(test_data.labels);
    
    % Sort each column by distance. This makes the first row represent
    % the best match for a given column (i.e. for a given test sample) and
    % the second row represents the second best match, etc...
    [~, rank_idx] = sort(dist);
    
    % Create a matrix of the labels that were matched to sorted in
    %   descending order by match distance
    % ex: matched_labels(N,M) is the label for the training sample
    %   which was the Nth closest match for test sample M
    matched_labels = training.labels(rank_idx);
    
    % Same size as matched_labels and contains the truth
    correct_labels = repmat(test_data.labels, length(training.labels), 1);
    
    % Compare results with truth
    correct_matches = (matched_labels == correct_labels);
    
    % Fill column below first match with ones. Used to generate the CMC
    % curve.
    correct_cum = cumsum(correct_matches, 1);
    correct_cum(correct_cum > 1) = 1;
    
    % Compute CMC by summing across each of the columns
    cmc = sum(correct_cum,2) / n_test_imgs;
    
    % Plot CMC results
    k = find(training.information >= args.information, 1, 'first');
    info = training.information(k);
    h = figure();
    plot(1:50, cmc(1:50), '-', 'Marker', '.', 'LineWidth', 2, 'MarkerSize', 20);
    title('Comparitive CMC graph');
    xlabel('Rank');
    ylabel('Performace');
    legend(sprintf('Information %0.2f', info), ...
        'Location', 'southeast');
    grid on;
    ax = axis();
    axis([ax(1) ax(2) ax(3) 1]); % always go up to 1
    % Save figure
    savefig(h, [args.resultsdir filesep 'CMC.fig']);
    print(h, [args.resultsdir filesep 'CMC.png'], '-dpng');
    
    % Save CMC to plot later
    save([args.resultsdir filesep 'CMC.mat'], 'cmc');
    
    % Get 3 correctly matched images (show both original and reconstructed)
    correct_idx = find(correct_cum(1,:), 3, 'first');
    correct_match_idx = rank_idx(1,correct_idx);

    correct1 = extract_image(test_data, correct_idx(1));
    correct1r = reshape(U*test_b(:,correct_idx(1)) + training.mean, training.img_size);
    correct2 = extract_image(test_data, correct_idx(2));
    correct2r = reshape(U*test_b(:,correct_idx(2)) + training.mean, training.img_size);
    correct3 = extract_image(test_data, correct_idx(3));
    correct3r = reshape(U*test_b(:,correct_idx(3)) + training.mean, training.img_size);

    correct_match_1 = extract_image(training, correct_match_idx(1));
    correct_match_1r = reshape(U*train_b(:,correct_match_idx(1)) + training.mean, training.img_size);
    correct_match_2 = extract_image(training, correct_match_idx(2));
    correct_match_2r = reshape(U*train_b(:,correct_match_idx(2)) + training.mean, training.img_size);
    correct_match_3 = extract_image(training, correct_match_idx(3));
    correct_match_3r = reshape(U*train_b(:,correct_match_idx(3)) + training.mean, training.img_size);

    h = figure();
    subplot(2,8,1);    imshow(correct1);           title('Correct 1');
    subplot(2,8,2);    imshow(correct1r);          title('Reconst.');
    subplot(2,8,4);    imshow(correct2);           title('Correct 2');
    subplot(2,8,5);    imshow(correct2r);          title('Reconst.');
    subplot(2,8,7);    imshow(correct3);           title('Correct 3');
    subplot(2,8,8);    imshow(correct3r);          title('Reconst.');
    subplot(2,8,9);    imshow(correct_match_1);    title('Match 1');
    subplot(2,8,10);   imshow(correct_match_1r);
    subplot(2,8,12);   imshow(correct_match_2);    title('Match 2');
    subplot(2,8,13);   imshow(correct_match_2r);
    subplot(2,8,15);   imshow(correct_match_3);    title('Match 3');
    subplot(2,8,16);   imshow(correct_match_3r);
    % Resize and save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) 930 285]);
    savefig(h, [args.resultsdir filesep 'CorrectMatches.fig']);
    print(h, [args.resultsdir filesep 'CorrectMatches.png'], '-dpng')
    
    % Get 3 incorrectly matched images (show both original and reconstructed)
    incorrect_idx = find(~correct_cum(1,:), 3, 'first');
    incorrect_match_idx = rank_idx(1,incorrect_idx);
    
    incorrect1 = extract_image(test_data, incorrect_idx(1));
    incorrect1r = reshape(U*test_b(:,incorrect_idx(1)) + training.mean, training.img_size);
    incorrect2 = extract_image(test_data, incorrect_idx(2));
    incorrect2r = reshape(U*test_b(:,incorrect_idx(2)) + training.mean, training.img_size);
    incorrect3 = extract_image(test_data, incorrect_idx(3));
    incorrect3r = reshape(U*test_b(:,incorrect_idx(3)) + training.mean, training.img_size);

    incorrect_match_1 = extract_image(training, incorrect_match_idx(1));
    incorrect_match_1r = reshape(U*train_b(:,incorrect_match_idx(1)) + training.mean, training.img_size);
    incorrect_match_2 = extract_image(training, incorrect_match_idx(2));
    incorrect_match_2r = reshape(U*train_b(:,incorrect_match_idx(2)) + training.mean, training.img_size);
    incorrect_match_3 = extract_image(training, incorrect_match_idx(3));
    incorrect_match_3r = reshape(U*train_b(:,incorrect_match_idx(3)) + training.mean, training.img_size);
    
    h = figure();
    subplot(2,8,1);    imshow(incorrect1);          title('Incorrect 1');
    subplot(2,8,2);    imshow(incorrect1r);         title('Reconst.');
    subplot(2,8,4);    imshow(incorrect2);          title('Incorrect 2');
    subplot(2,8,5);    imshow(incorrect2r);         title('Reconst.');
    subplot(2,8,7);    imshow(incorrect3);          title('Incorrect 3');
    subplot(2,8,8);    imshow(incorrect3r);         title('Reconst.');
    subplot(2,8,9);    imshow(incorrect_match_1);   title('Match 1');
    subplot(2,8,10);   imshow(incorrect_match_1r);
    subplot(2,8,12);   imshow(incorrect_match_2);   title('Match 2');
    subplot(2,8,13);   imshow(incorrect_match_2r);
    subplot(2,8,15);   imshow(incorrect_match_3);   title('Match 3');
    subplot(2,8,16);   imshow(incorrect_match_3r);
    % Resize and save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) 930 285]);
    savefig(h, [args.resultsdir filesep 'IncorrectMatches.fig']);
    print(h, [args.resultsdir filesep 'IncorrectMatches.png'], '-dpng')
end
