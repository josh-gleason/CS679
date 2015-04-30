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
        
        if args.numfeats == 2 && args.experiment <= 4
            figure(10+idx); hold off;
            plot(training(1,labels == 1), training(2,labels == 1), ' *b'); hold on;
            plot(training(1,labels == 2), training(2,labels == 2), ' *r'); hold on;
            
            ax = axis();
            xmin = ax(1);            xmax = ax(2);
            ymin = ax(3);            ymax = ax(4);
            
            [x,y] = meshgrid(linspace(xmin,xmax,500), linspace(ymin,ymax,500));
            v = [x(:) y(:)].';
            
            W1 = -1/2 * inv(model{idx}.sigma1);
            w1 = model{idx}.sigma1\model{idx}.mu1;
            w10 = -1/2 * model{idx}.mu1.' * w1 - 1/2 * log(det(model{idx}.sigma1)) + log(0.5);

            g1 = sum(v .* (W1 * v)) + sum(bsxfun(@times, v, w1)) + w10;

            W2 = -1/2 * inv(model{idx}.sigma2);
            w2 = model{idx}.sigma1\model{idx}.mu2;
            w20 = -1/2 * model{idx}.mu2.' * w2 - 1/2 * log(det(model{idx}.sigma2)) + log(0.5);

            g2 = sum(v .* (W2 * v)) + sum(bsxfun(@times, v, w2)) + w20;
            
            prediction = ones(500*500,1);
            prediction(g2 > g1) = 2;
            prediction = reshape(prediction, [500,500]);
            hold off;
            colormap([0.7 0.7 0.9; 0.9 0.7 0.7]);
            contourf(x,y,prediction,[1.5]); hold on;
            plot(training(1,labels == 1), training(2,labels == 1), ' *b'); hold on;
            plot(training(1,labels == 2), training(2,labels == 2), ' *r'); hold on;
            axis(ax);
            xlabel('Feature 1');
            ylabel('Feature 2');
            res = 'low res';
            if args.resolution == 2
                res = 'high res';
            end
            title(sprintf('Fold %d Decision Boundary (%s)\nBayesian', idx, res));
        end
    end
    if args.experiment <= 4
        fprintf('Saving training results to %s\n', args.trainingfile);
        save(args.trainingfile, 'model');
    end
end

