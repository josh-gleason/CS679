function training = load_training(args)
%LOAD_TRAINING Load training data, see train_eigenface for structure of
% training

    fprintf('Loading training data from %s\n ...\n', args.trainingfile);
    load(args.trainingfile);
    fprintf('Complete\n');

    if ~exist('training', 'var')
        error('Training data not found in %s', args.trainingfile);
    end 
end

