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
end