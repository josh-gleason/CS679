function model = train_2(args)
%TRAIN_2 Train the Bayesian classifier (assuming Gaussian)

    train_feats_fnames = {'trPCA_01.txt', ...
                          'trPCA_02.txt', ...
                          'trPCA_03.txt'};
    train_label_fnames = {'TtrPCA_01.txt', ...
                          'TtrPCA_02.txt', ...
                          'TtrPCA_03.txt'};
    FOLDS = 3;

    model = cell(1, FOLDS);
    % Train each fold seperately
    for idx = 1:FOLDS
        training = dlmread([args.datadir filesep train_feats_fnames{idx}], ' ');
        labels = dlmread([args.datadir filesep train_label_fnames{idx}], ' ');
        
        training = training(1:args.numfeats, :);
        
        % Seperate into two classes
        training_1 = training(:, labels == 1);
        training_2 = training(:, labels == 2);
        
        [~, samples_1] = size(training_1);
        [~, samples_2] = size(training_2);
        
        % Maximum likelihood estimate of model
        model{idx}.mu1 = sum(training_1, 2) / samples_1;
        model{idx}.mu2 = sum(training_2, 2) / samples_2;
        
        samples0_1 = bsxfun(@minus, training_1, model{idx}.mu1);
        samples0_2 = bsxfun(@minus, training_2, model{idx}.mu2);
        
        % estimate covariance
        model{idx}.sigma1 = (samples0_1 * samples0_1.') / (samples_1 - 1);
        model{idx}.sigma2 = (samples0_2 * samples0_2.') / (samples_2 - 1);
    end
    if args.experiment <= 4
        fprintf('Saving training results to %s\n', args.trainingfile);
        save(args.trainingfile, 'model');
    end
end

