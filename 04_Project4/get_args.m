function args = get_args(arglist)
%GET_ARGS Parse the arguments passed to main and query user for any missing
% arguments

    args.experiment = -1;
    args.resolution = -1;
    args.datadir = 'data';
    args.resultsdir = 'results';
    args.trainingfile = 'auto';
    args.numfeats = -1;
    % Only used for SVM classifier
    args.classifier_params = struct('kernel',-1, ...
                                    'gamma' ,-1, ...
                                    'coef0' ,-1, ...
                                    'degree',-1, ...
                                    'cost'  ,-1);

    NUM_FEATS = 134;
            
    SVM_CLASSIFIER   = 1;
    BAYES_CLASSIFIER = 2;
    
    LOW_RES  = 1;
    HIGH_RES = 2;
    
    SVM_KERNEL_POLY = 1;
    SVM_KERNEL_RBF  = 2;
    
    DEF_FEATS = 30;
    MAX_FEATS = 134;
    
    % Parse arguments
    idx = 1;
    while idx < length(arglist)
        if ~ischar(arglist{idx})
            error('Expected string argument identifier');
        end
        switch lower(arglist{idx})
            case {'classifier', 'c'}
                args.classifier = arglist{idx+1};
                if ~isfloat(args.experiment)
                    args.experiment = -1;
                end
            case {'experiment', 'exp'}
                args.experiment = arglist{idx+1};
                if ~isfloat(args.experiment)
                    args.experiment = -1;
                end
            case {'resolution', 'res'}
                args.resolution = arglist{idx+1};
                if ~isfloat(args.resolution)
                    args.resolution = -1;
                end
            case {'kernel', 'k'}
                args.classifier_params.kernel = arglist{idx+1};
                if ~isfloat(args.classifier_params.kernel)
                    args.classifier_params.kernel = -1;
                end
            case {'gamma', 'g'}
                args.classifier_params.gamma = arglist{idx+1};
                if ~isfloat(args.classifier_params.gamma);
                    args.classifier_params.gamma = -1;
                end
            case {'coef0', 'c0'}
                args.classifier_params.coef0 = arglist{idx+1};
                if ~isfloat(args.classifier_params.coef0)
                    args.classifier_params.coef0 = -1;
                end
            case {'degree', 'd'}
                args.classifier_params.degree = arglist{idx+1};
                if ~isfloat(args.classifier_params.degree)
                    args.classifier_params.degree = -1;
                end
            case {'cost'}
                args.classifier_params.cost = arglist{idx+1};
                if ~isfloat(args.classifier_params.cost)
                    args.classifier_params.cost = -1;
                end
            case {'nfeats', 'f'}
                args.numfeats = arglist{idx+1};
                if ~isfloat(args.numfeats)
                    args.classifier_params.cost = -1;
                end
            case {'datadir', 'dd'}
                % Data directory (string)
                args.datadir = arglist{idx+1};
                if ~ischar(datadir)
                    args.datadir = '';
                end
            case {'resultsdir', 'rd'}
                % Results directory (string)
                args.resultsdir = arglist{idx+1};
                if ~ischar(resultsdir)
                    args.resultsdir = '';
                end
            case {'trainingfile', 'training', 'train'}
                % Explicitly request training file
                args.trainingfile = arglist{idx+1};
                if ~ischar(args.trainingfile)
                    args.trainingfile = '';
                end
            otherwise
                error('Unrecognized parameter option %s\n', arglist{idx});
        end
        idx = idx + 2;
    end

    % Query user for missing arguments
    while ~(args.experiment == 1 || args.experiment == 2 || args.experiment == 3 || args.experiment == 4 || args.experiment == 5 || args.experiment == 6)
        fprintf('Which Experiment would you like to run?\n');
        fprintf('  1. Train experiment 1 (SVM Classification)\n');
        fprintf('  2. Train experiment 2 (Bayesian Classification)\n');
        fprintf('  3. Run experiment 1 (SVM Classification)\n');
        fprintf('  4. Run experiment 2 (Bayesian Classification)\n');
        fprintf('  5. Train/Test Experiment 1 (SVM Classification) no file output\n');
        fprintf('  6. Train/Test Experiment 2 (Bayes Classification) no file output\n');
        args.experiment = input('Enter Selection (Default 1): ');
        if isempty(args.experiment)
            args.experiment = 1;
        end
        if ~isfloat(args.experiment)
            args.experiment = -1;
        end
    end
    if args.experiment == 1 || args.experiment == 3 || args.experiment == 5
        args.classifier = SVM_CLASSIFIER;
    else
        args.classifier = BAYES_CLASSIFIER;
    end

    % Functions for querying other arguments from user if not provided
    function query_resolution()
        while ~(args.resolution == HIGH_RES || args.resolution == LOW_RES)
            fprintf('Which resolution would you like to experiment with?\n');
            fprintf('  %d. Low resolution\n', LOW_RES);
            fprintf('  %d. High resolution\n', HIGH_RES);
            args.resolution = input(sprintf('Enter Selection (Default %d): ', LOW_RES));
            if isempty(args.resolution)
                args.resolution = LOW_RES;
            end
            if ~isfloat(args.resolution)
                args.resolution = -1;
            end
            args.resolution = round(args.resolution);
        end
    end

    function query_numfeats()
        while ~(args.numfeats <= MAX_FEATS && args.numfeats > 0 && round(args.numfeats) == args.numfeats)
            fprintf('How many features would you like to use? (Max : %d)\n', MAX_FEATS);
            args.numfeats = input(sprintf('Enter number (Default %d): ', DEF_FEATS));
            if isempty(args.numfeats)
                args.numfeats = DEF_FEATS;
            end
            if ~isfloat(args.numfeats)
                args.numfeats = -1;
            end
        end
    end

%     function query_fold()
%         while ~(args.fold == 1 || args.fold == 2 || args.fold == 3)
%             fprintf('Which fold would you like to use?\n');
%             fprintf('  1. Fold 1\n');
%             fprintf('  2. Fold 2\n');
%             fprintf('  3. Fold 3\n');
%             args.fold = input('Enter Selection (Default 1): ');
%             if isempty(args.fold)
%                 args.fold = 1;
%             end
%             if ~isfloat(args.fold)
%                 args.fold = -1;
%             end
%         end
%     end

    function query_datadir()
        while isempty(args.datadir) || ~isdir(args.datadir)
            args.datadir = uigetdir('.', 'Data Directory');
        end
    end

    function query_svn_params()
        while ~(args.classifier_params.kernel == SVM_KERNEL_POLY || args.classifier_params.kernel == SVM_KERNEL_RBF)
            fprintf('Select a Kernel function\n');
            fprintf('  %d. Polynomial Kernel\n', SVM_KERNEL_POLY);
            fprintf('  %d. RBF Kernel\n', SVM_KERNEL_RBF);
            args.classifier_params.kernel = input(sprintf('Enter Selection (Default %d) : ', SVM_KERNEL_POLY));
            if isempty(args.classifier_params.kernel)
                args.classifier_params.kernel = SVM_KERNEL_POLY;
            end
        end
        
        while args.classifier_params.cost < 0
            args.classifier_params.cost = input('Enter cost      (Default 1) : ');
            if isempty(args.classifier_params.cost)
                args.classifier_params.cost = 10;
            end
        end
        while args.classifier_params.gamma < 0
            args.classifier_params.gamma = input(sprintf('Enter gamma (Default %0.6f) : ', 1/NUM_FEATS));
            if isempty(args.classifier_params.gamma)
                args.classifier_params.gamma = 1/NUM_FEATS;
            end
        end
        if args.classifier_params.kernel == SVM_KERNEL_POLY
            while args.classifier_params.coef0 < 0
                args.classifier_params.coef0  = input('Enter coef0     (Default 0) : ');
                if isempty(args.classifier_params.coef0)
                    args.classifier_params.coef0 = 0;
                end
            end
            while args.classifier_params.degree < 0
                args.classifier_params.degree = input('Enter degree    (Default 3) : ');
                if isempty(args.classifier_params.degree)
                    args.classifier_params.degree = 3;
                end
            end
        end
        while ~isfield(args, 'classifier_params')
            fprintf('Enter SVM parameters');
        end
    end

    function filename = gen_dir()
        switch args.classifier
            case SVM_CLASSIFIER
                filename = 'SVM';
            case BAYES_CLASSIFIER
                filename = 'BAYES';
        end
        switch args.resolution
            case HIGH_RES
                filename = [filename '_H'];
            case LOW_RES
                filename = [filename '_L'];
        end
        
        %filename = [filename '_' num2str(args.fold)];
        
        switch args.classifier
            case SVM_CLASSIFIER
                switch args.classifier_params.kernel
                    case SVM_KERNEL_POLY
                        filename = [filename sprintf('_%.3g_%.3g_%.3g', ...
                                                     args.classifier_params.gamma, ...
                                                     args.classifier_params.coef0, ...
                                                     args.classifier_params.degree)];
                    case SVM_KERNEL_RBF
                        filename = [filename sprintf('_%.3g',args.classifier_params.gamma)];
                end
                filename = [filename sprintf('_%.3g',args.classifier_params.cost)];
            case BAYES_CLASSIFIER
                % no parameters
        end
        filename = [filename '_' datestr(now,'yyyymmddHHMMSSFFF')];
    end

    function filename = gen_trainingdir()
        % generate training directory
        filename = ['training_' gen_dir()];
    end

    function filename = gen_resultsdir()
        filename = ['results' filesep 'experiment_' gen_dir()];
        if ~exist(filename, 'dir')
            mkdir(filename)
        end
    end

    function query_trainingfile()
        % attempt to automatically find the most recent training file with the correct resolution
        % All other arguments must be set before calling this
        if strcmp(args.trainingfile,'auto')
            d = dir(args.resultsdir);
            dir_names = {d([d.isdir]).name};
            
            % Find training data of the same type
            % Format of training file name
            % training_<SVM/BAYES>_<H/L>_<1/2/3>_<PARAMS>_<DATESTAMP>.mat
            % where params holds the SVM kernel type and parameters
            valid_idx = [];
            valid_times = [];
            for dir_idx = 1:length(dir_names)
                % Check to see if the training directory is valid based on training parameters
                dirname = dir_names{dir_idx};
                training_dir = gen_trainingdir();
                % remove timestamp from filename
                str1 = training_dir(1:find(training_dir=='_',1,'last')-1);
                str2 = dirname(1:find(dirname=='_',1,'last')-1);
                if strcmp(str1,str2);
                    timestamp = dirname(find(dirname=='_',1,'last')+1:end);
                    valid_idx(end+1) = dir_idx;
                    valid_times(end+1) = datenum(timestamp, 'yyyymmddHHMMSSFFF');
                end
            end
            if ~isempty(valid_idx)
                % Retrieve the most recent file
                [~,v] = max(valid_times);
                args.trainingfile = [args.resultsdir filesep d(valid_idx(v)).name filesep 'training.mat'];
            else
                % will query
                fprintf('No matching training sets found\n');
                args.trainingfile = '';
            end
        end
        % If auto not used or fails and designated file doesn't exist then query the user
        while ~(exist(args.trainingfile, 'file')==2)
            [fname, fpath] = uigetfile([args.resultsdir filesep '*.mat'], 'Select Training File');
            % This means the dialog was canceled
            if ~ischar(fname)
                error('No file selected');
            end
            args.trainingfile = [fpath filesep fname];
        end
        
        fprintf(1, 'Using training file %s\n', args.trainingfile);
    end

    query_datadir();
    query_resolution();
    query_numfeats();
    if args.experiment == 1 || args.experiment == 3 || args.experiment == 5
        query_svn_params();
    end
    if args.experiment == 3 || args.experiment == 4
        query_trainingfile();
        args.resultsdir = gen_resultsdir();
    elseif args.experiment == 1 || args.experiment == 2
        traindir = [args.resultsdir filesep gen_trainingdir()];
        args.trainingfile = [traindir filesep 'training.mat'];
        if ~exist(traindir, 'dir')
            mkdir(traindir);
        end
    end
    if args.resolution == LOW_RES
        args.datadir = [args.datadir filesep '16_20'];
    else
        args.datadir = [args.datadir filesep '48_60'];
    end
    args.trainingfile
end

