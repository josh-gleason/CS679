function experiment_2(args)

    % Load training data
    training = load_training(args);
    
    % Load test images
    test_data = load_test_data(args.testdir, training.mean);
    
    % Compare each test sample with each training sample in eigenspace
    dist = compare_samples(training, test_data, args.information);
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Data available at this point in the code
    %
    %   training
    %       labels      : 1xN list of labels identifying each of the
    %                       samples in the training data.
    %       mean        : Dx1 mean face (linearized)
    %       samples     : DxN set of samples WITH MEAN FACE SUBTRACTED
    %                       Each column is a single sample.
    %       img_size    : Contains the size of the original image.
    %       information : 1xN cumulative sum of the information retained by
    %                      the top principal components.
    %                      ex. information(k) is the amount of information
    %                      retained if the top k eigenvectors are kept.
    %       eigenvectors: DxN unit column eigenvectors sorted left to right
    %                       by decreasing eigenvector magnitude.
    %       eigenvalues : 1xN list of eigenvalues corresponding to each
    %                       eigenvector
    %   
    %   test_data
    %       labels  : 1xM list of labels for each of the test samples
    %       mean    : The same as training.mean
    %       samples : DxM set of samples WITH MEAN FACE SUBTRACTED. Each
    %                   column is a single sample.
    %       img_size: The same as training.img_size
    %
    %   dist: NxM matrix
    %       Contains the Mahalanobis distance between each test sample
    %       and each training sample (compared in lower dimensional space)
    %       Ex: dist(A,B) is the Mahalanobis distance between training
    %       sample A and test sample B.
    %
    %   args
    %       information   : Ratio of information to retain
    %       resultsdir    : A timestamped directory created for the results
    %                         of this experiment.
    %       trainingfile  : The filename that training was loaded from
    %                         (should have been determined automatically).
    %       removesubjects: The number of training subjects removed during
    %                         the training phase. Only for reference since
    %                         the subjects been removed by the time this
    %                         experiment starts.
    %
    % Note: By the time the experiment starts a timestamped results
    % directory has already been created at args.resultsdir
    % All results should be saved to this directory.
    %
    % -Josh
    
    % Experiment B ...
      
    trainLabels = repmat(training.labels',1,length(test_data.labels),1);
    testLabels  = repmat(test_data.labels,length(training.labels),1);
    
    matchSamples  = trainLabels == testLabels;
    colsIntruder   = sum(matchSamples,1) == 0;
    
    maxDm = max(dist(:));
    dx = maxDm / 25;
    domainX = 0:dx:maxDm;
        
    hist_w0 = hist(dist(matchSamples == 1),domainX);
    hist_w1 = hist(dist(:,colsIntruder),domainX);
    
    nSamples_w0 = sum(hist_w0);
    nSamples_w1 = sum(hist_w1);
   
    pdf_w0 = hist_w0/nSamples_w0;
    pdf_w1 = hist_w1/nSamples_w1;
    
    cdf_w0 = cumsum(hist_w0)/nSamples_w0;
    cdf_w1 = cumsum(hist_w1)/nSamples_w1;
   
    subplot(2,2,1);
    h=figure; hold on;
    plot(domainX,pdf_w0,'r',domainX,pdf_w1,'b'); grid on;
    title('p(d_M | w_0) and p(d_M | w_1)');
    legend('w_0 = friendly', 'w_1=intruder');
    xlabel('d_M (Mahalanobis distance)');
    ylabel('probability density');
    
    subplot(2,2,2);    
    plot(domainX,cdf_w0,'r',domainX,cdf_w1,'b'); grid on;
    title('P(d_M | w_0) and P(d_M | w_1)');
    legend('w_0 = friendly', 'w_1=intruder');
    xlabel('d_M (Mahalanobis distance)');
    ylabel('cumulative probability density');

    subplot(2,2,3);    
    figure; plot(cdf_w1,cdf_w0,'b');
    title('ROC: p(d_M | w_0) versus p(d_M | w_1)');
    legend('w_0 = friendly', 'w_1=intruder');
    xlabel('p(d_M | w_1)');
    ylabel('p(d_M | w_0)');
   
    % Resize and save figure
    p = get(h, 'Position');
    set(h, 'Position', [p(1) p(2) 930 285]);
    savefig(h, [args.resultsdir filesep 'PartB_Performance.fig']);
    print(h, [args.resultsdir filesep 'PartB_Performance.png'], '-dpng')
    
end