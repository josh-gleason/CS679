function accuracy = test_1(args, model_opt)
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
        test_feats = dlmread([args.datadir filesep test_feats_fnames{idx}], ' ').';
        test_labels = dlmread([args.datadir filesep test_label_fnames{idx}], ' ').';
        
        val_feats = dlmread([args.datadir filesep val_feats_fnames{idx}], ' ').';
        val_labels = dlmread([args.datadir filesep val_label_fnames{idx}], ' ').';
        
        feats = [test_feats(:, 1:args.numfeats); val_feats(:, 1:args.numfeats)];
        labels = [test_labels; val_labels];
        fprintf('Predicting Fold %d...\n', idx);
        [prediction, svmaccuracy, ~] = svmpredict(labels, feats, model{idx});
        fprintf('Complete\n\n');
        
        predictions{idx} = prediction;
        correct{idx} = (prediction == labels);
        accuracy(idx) = svmaccuracy(1);
    end
    
    if save_results
        resultsfile = [args.resultsdir filesep 'results.mat'];
        fprintf('Saving results to %s\n', resultsfile);
        save(resultsfile, 'predictions', 'correct', 'accuracy');
    end
end

