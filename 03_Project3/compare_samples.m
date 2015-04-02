function [dist, U, train_b, test_b] = compare_samples(training, test_data, information)
%COMPARE_SAMPLES Compare the test_data samples to the training data samples
% using PCA. The dist matrix returned is a N x M matrix where N is the
% number of training samples and M is the number of test samples. The value
% at dist(n,m) is the comparison between training sample n and test sample
% m

    % Choose the top eigenvectors preserving some set amount of information
    k = find(training.information >= information, 1, 'first');
    U = training.eigenvectors(:,1:k);

    % Compute the covariance 
    covariance = U.'*training.samples*training.samples.'*U;
    
    % Project both training and testing to eigenspace
    train_b = U.' * training.samples;
    test_b = U.' * test_data.samples;
    
    [~,n_train_imgs] = size(train_b);
    [~,n_test_imgs] = size(test_b);
    
    % calculate Mahalanobis distance between each training and test image
    dist = zeros(n_train_imgs, n_test_imgs);
    for idx = 1:n_train_imgs
        % subtract train_b(:,idx) from each test_b
        db = bsxfun(@minus, test_b, train_b(:,idx));
        dist(idx,:) = sqrt(sum(db .* (covariance \ db), 1));
    end
    
end

