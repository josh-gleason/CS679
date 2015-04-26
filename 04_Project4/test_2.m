function accuracy = test_2(args, model_opt)
%TEST_2 Test the Bayesian classifier
    %TEST_1 Test SVM classifier
    test_feats_fnames = {'tsPCA_01.txt', ...
                         'tsPCA_02.txt', ...
                         'tsPCA_03.txt'};
    val_feats_fnames = {'valPCA_01.txt', ...
                        'valPCA_02.txt', ...
                        'valPCA_03.txt'};
    test_label_fnames = {'TtsPCA_01.txt', ...
                         'TtsPCA_02.txt', ...
                         'TtsPCA_03.txt'};
	val_label_fnames = {'TvalPCA_01.txt', ...
                        'TvalPCA_02.txt', ...
                        'TvalPCA_03.txt'};

    FOLDS = 3;

    % Loads classifier model
    save_results = 1;
    if exist('model_opt', 'var')
        model = model_opt;
        save_results = 0;
    else
        load(args.trainingfile);
        assert(exist('model', 'var')==1);
    end
    
    predictions = cell(1, FOLDS);
    correct = cell(1, FOLDS);
    accuracy = zeros(1, FOLDS);
    
    for idx = 1:FOLDS
        test_feats = dlmread([args.datadir filesep test_feats_fnames{idx}], ' ');
        test_labels = dlmread([args.datadir filesep test_label_fnames{idx}], ' ');
        
        val_feats = dlmread([args.datadir filesep val_feats_fnames{idx}], ' ');
        val_labels = dlmread([args.datadir filesep val_label_fnames{idx}], ' ');
        
        feats = [test_feats val_feats];
        labels = [test_labels val_labels];
        
        feats = feats(1:args.numfeats, :);
        
        % Compute discriminate functions for each feature (column)
        W1 = -1/2 * inv(model{idx}.sigma1);
        w1 = model{idx}.sigma1\model{idx}.mu1;
        w10 = -1/2 * model{idx}.mu1.' * w1 - 1/2 * log(det(model{idx}.sigma1)) + log(0.5);
        
        g1 = sum(feats .* (W1 * feats)) + sum(bsxfun(@times, feats, w1)) + w10;
        
        W2 = -1/2 * inv(model{idx}.sigma2);
        w2 = model{idx}.sigma1\model{idx}.mu2;
        w20 = -1/2 * model{idx}.mu2.' * w2 - 1/2 * log(det(model{idx}.sigma2)) + log(0.5);
        
        g2 = sum(feats .* (W2 * feats)) + sum(bsxfun(@times, feats, w2)) + w20;
        
        % Make prediction using discriminate functions
        prediction = ones(size(labels));
        prediction(g2 > g1) = 2;
        predictions{idx} = prediction;
        
        correct{idx} = (prediction == labels);
        accuracy(idx) = 100.0 * sum(correct{idx}) / length(labels);
    end

    if save_results
        resultsfile = [args.resultsdir filesep 'results.mat'];
        fprintf('Saving results to %s\n', resultsfile);
        save(resultsfile, 'predictions', 'correct', 'accuracy');
    end
end
