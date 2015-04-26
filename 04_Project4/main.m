function retval = main(varargin)
    args = get_args(varargin);
    retval = [];

    addpath('libsvm');

    switch args.experiment
        case 1
            train_1(args);
        case 2
            train_2(args);
        case 3
            test_1(args);
        case 4
            test_2(args);
        case 5
            % SVM train/test
            model = train_1(args);
            retval = test_1(args, model);
        case 6
            % Bayesian train/test
            model = train_2(args);
            retval = test_2(args, model);
    end
end