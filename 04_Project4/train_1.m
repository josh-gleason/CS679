function model = train_1(args)
%TRAIN_1 Train SVM classifier
    train_feats_fnames = {'trPCA_01.txt', ...
                          'trPCA_02.txt', ...
                          'trPCA_03.txt'};
    train_label_fnames = {'TtrPCA_01.txt', ...
                          'TtrPCA_02.txt', ...
                          'TtrPCA_03.txt'};
    FOLDS = 3;

    SVM_KERNEL_POLY = 1;
    SVM_KERNEL_RBF  = 2;
    
    model = cell(1, FOLDS);
    % Train each fold seperately
    for idx = 1:FOLDS
        training = dlmread([args.datadir filesep train_feats_fnames{idx}], ' ').';
        labels = dlmread([args.datadir filesep train_label_fnames{idx}], ' ').';

        training = training(:, 1:args.numfeats);
        
        if args.classifier_params.kernel == SVM_KERNEL_POLY
            svm_options = sprintf('-t 1 -d %0.6f -g %0.6f -r %0.6f -c %0.6f', ...
                    args.classifier_params.degree, ...
                    args.classifier_params.gamma, ...
                    args.classifier_params.coef0, ...
                    args.classifier_params.cost);
        elseif args.classifier_params.kernel == SVM_KERNEL_RBF
            svm_options = sprintf('-t 2 -g %0.6f -c %0.6f', ...
                    args.classifier_params.gamma, ...
                    args.classifier_params.cost);
        end

        fprintf('Training Classifier %d/%d\n', idx, FOLDS);
        model{idx} = svmtrain(labels, training, svm_options);
        fprintf('Complete\n\n');
        
    end
    if args.experiment <= 4
        fprintf('Saving training results to %s\n', args.trainingfile);
        save(args.trainingfile, 'model');
    end
end
