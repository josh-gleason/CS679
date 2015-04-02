function training(args)
%training Train and test the eigenface reconstruction on the first face in the
% test data and make sure it reconstructs correctly. Also display the mean
% face and top 10 and bottom 10 eigenfaces
    training = train_eigenface(args);

    % Reconstruct first image in training data
    image = reshape(training.samples(:,1) + training.mean, training.img_size);
    b = training.eigenvectors.'*training.samples(:,1);
    image_rep = reshape(training.eigenvectors*b + training.mean, training.img_size);

    % Show reconstructed image
    h = figure();
    subplot(1,2,1);
    imshow(image);
    title('Original');
    subplot(1,2,2);
    imshow(image_rep);
    title('Reconstructed');
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) p(3) 285]);
    % Save figure
    savefig(h, [args.resultsdir filesep 'reconstruction.fig']);
    print(h, [args.resultsdir filesep 'reconstruction.png'], '-dpng');

    % Experiment a.I
    % Show the average face
    h = figure();
    img_mean = reshape(training.mean, training.img_size);
    img_mean = imresize(img_mean, 4, 'nearest');
    imshow(img_mean);
    title('Mean Face');
    % Save figure
    savefig(h, [args.resultsdir filesep 'mean.fig']);
    print(h, [args.resultsdir filesep 'mean.png'], '-dpng');

    % Show the top 10 largest Eigenfaces
    h = figure();
    for idx = 1:10
        img_eigface = reshape(training.eigenvectors(:,idx), training.img_size);
        img_eigface = (img_eigface - min(img_eigface(:))) / (max(img_eigface(:)) - min(img_eigface(:)));
        subplot(2,5,idx);
        imshow(img_eigface);
        title(sprintf('%d', idx));
    end
    % Save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) p(3) 285]);
    savefig(h, [args.resultsdir filesep 'top10.fig']);
    print(h, [args.resultsdir filesep 'top10.png'], '-dpng');

    % Show the bottom 10 largest Eigenfaces
    h = figure();
    for idx = 1:10
        img_eigface = reshape(training.eigenvectors(:,end-(idx+1)), training.img_size);
        img_eigface = (img_eigface - min(img_eigface(:))) / (max(img_eigface(:)) - min(img_eigface(:)));
        subplot(2,5,idx);
        imshow(img_eigface);
        title(sprintf('%d', idx));
    end
    % Save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) p(3) 285]);
    savefig(h, [args.resultsdir filesep 'bot10.fig']);
    print(h, [args.resultsdir filesep 'bot10.png'], '-dpng');
    
    % Compute reprojection error
    err = norm(image(:) - image_rep(:),2);

    % Save training
    save(args.trainingfile, 'training');
    
    % Save results
    results_fname = [args.resultsdir filesep 'results.txt'];
    fresults = fopen(results_fname, 'w');
    fprintf(fresults, 'Reprojection Error: %0.4f\n', err);
    fclose(fresults);
    type(results_fname); % echo results file to terminal
end