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

        % If using two classes then plot results
        if args.numfeats == 2 && args.experiment <= 4
            figure(10+idx); hold off;
            plot(training(labels == 1,1), training(labels == 1,2), ' *b'); hold on;
            plot(training(labels == 2,1), training(labels == 2,2), ' *r'); hold on;
            
            ax = axis();
            xmin = ax(1);            xmax = ax(2);
            ymin = ax(3);            ymax = ax(4);
            
            [x,y] = meshgrid(linspace(xmin,xmax,500), linspace(ymin,ymax,500));
            v = [x(:) y(:)];
            prediction = svmpredict(zeros(500*500,1), v, model{idx});
            prediction = reshape(prediction, [500,500]);
            hold off;
            colormap([0.7 0.7 0.9; 0.9 0.7 0.7]);
            contourf(x,y,prediction,[1.5]); hold on;
            plot(training(labels == 1,1), training(labels == 1,2), ' *b'); hold on;
            plot(training(labels == 2,1), training(labels == 2,2), ' *r'); hold on;
            plot(training(model{idx}.sv_indices, 1), training(model{idx}.sv_indices, 2), ' ok', 'MarkerSize', 8, 'LineWidth', 2);
            axis(ax);
            xlabel('Feature 1');
            ylabel('Feature 2');
            res = 'low res';
            if args.resolution == 2
                res = 'high res';
            end
            title(sprintf('Fold %d Decision Boundary (%s)\nParams (%s)', idx, res, svm_options));
        end
        
    end
    if args.experiment <= 4
        fprintf('Saving training results to %s\n', args.trainingfile);
        save(args.trainingfile, 'model');
    end
end
