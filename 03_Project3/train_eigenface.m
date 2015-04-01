function training = train_eigenface(args)
%TRAIN_EIGENFACE Train Eigenface

    % Load samples
    [samples, labels, img_size] = load_dataset(args.traindir);
    
    % Calculate mean face and subtract
    mean_face = mean(samples, 2);
    samples = bsxfun(@minus, samples, mean_face);

    % Depending on the size of A decide if we want to use eigenvalues and
    % eigenvectors from AA' or A'A (For low resolution images we want to
    % use AA' for high res use A'A
    [dims, n_samples] = size(samples);
    if dims > n_samples
        % In this case the dimensionality is very high so use A'A
        cov = samples.' * samples;
        [eigenvectors, eigenvalues] = eig(cov);
        % Organize by decreasing eigenvalue order
        eigenvectors = fliplr(eigenvectors);
        eigenvalues = fliplr(diag(eigenvalues).');
        
        % Transform eigenvectors into correct space and normalize
        eigenvectors = samples * eigenvectors;
        eigenvectors = eigenvectors ./ repmat(sqrt(sum(eigenvectors.^2)), [dims, 1]);
    else
        % In this case the dimensionality is not so high so use AA'
        cov = samples * samples.';
        [eigenvectors, eigenvalues] = eig(cov);
        % Organize by decreasing eigenvalue order
        eigenvectors = fliplr(eigenvectors);
        eigenvalues = fliplr(diag(eigenvalues).');
    end 
    
    % Calculate the sum of all the eigenvalues
    total = sum(eigenvalues);
    % Create a cumulative function to find the right amount of information to retain
    information = cumsum(eigenvalues) / total;
    
    % Save training data
    training.d = dims;
    training.labels = labels;
    training.mean = mean_face;
    training.samples = samples;
    training.img_size = img_size;
    training.information = information;
    training.eigenvectors = eigenvectors;
    training.eigenvalues = eigenvalues;
    training.covariance = samples * samples.';
end

