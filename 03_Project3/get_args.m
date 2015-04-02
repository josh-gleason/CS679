function args = get_args(arglist)
%PARSE_ARGS Parse the arguments for main, returns a struct with the
% paramters loaded.
    % Defaults
    args.experiment = -1;
    args.testset = -1;
    args.information = -1;
    args.removesamples = -1;
    args.datadir = 'data';
    args.resultsdir = 'results';
    args.trainingfile = 'auto';

    testset_str = {'H', 'L'};
    
    % Parse arguments
    idx = 1;
    while idx < length(arglist)
        if ~ischar(arglist{idx})
            error('Expected string');
        end
        switch lower(arglist{idx})
            case {'experiment', 'exp'}
                % 0, 1, or 2
                args.experiment = arglist{idx+1};
                if ~isfloat(args.experiment)
                    args.experiment = -1;
                end
            case {'testset', 'set'}
                % Valid options {'high', 'h'}, {'low', 'l'}
                args.testset = arglist{idx+1};
                if ~isfloat(args.testset)
                    args.testset = -1;
                end
            case {'information', 'info'}
                % Valid values 1 <= i <= 0
                args.information = arglist{idx+1};
                if ~isfloat(args.information)
                    args.information = -1;
                end
            case {'datadir', 'dd'}
                % Data directory
                args.datadir = arglist{idx+1};
                if ~ischar(datadir)
                    args.datadir = '';
                end
            case {'resultsdir', 'rd'}
                % Results directory
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
            case {'removesamples', 'rm'}
                args.removesamples = arglist{idx+1};
                if ~isfloat(args.removesamples)
                    args.removesamples = -1;
                end
                args.removesamples = round(args.removesamples);
            otherwise
                error('Unrecognized parameter option %s\n', arglist{idx});
        end
        idx = idx + 2;
    end
    
    % Query user for missing arguments
    while ~(args.experiment == 0 || args.experiment == 1 || args.experiment == 2)
        fprintf('Which Experiment would you like to run?\n');
        fprintf('  0. Train and test reconstruction (also run experiment a.I)\n');
        fprintf('  1. Run experiments A.II, A.III, and A.IV\n');
        fprintf('  2. Run experiment B\n');
        args.experiment = input('Enter Selection (Default 0): ');
        if isempty(args.experiment)
            args.experiment = 0;
        end
        if ~isfloat(args.experiment)
            args.experiment = -1;
        end
        args.experiment = round(args.experiment);
    end
    
    % Functions for querying other arguments from user if not provided
    function query_dataset()
        while ~(args.testset == 1 || args.testset == 2)
            fprintf('Which Test Set would you like to use?\n');
            fprintf('  1. High Resolution\n');
            fprintf('  2. Low Resolution\n');
            args.testset = input('Enter Selection (Default 1): ');
            if isempty(args.testset)
                args.testset = 1;
            end
            if ~isfloat(args.testset)
                args.testset = -1;
            end
            args.testset = round(args.testset);
        end
    end

    function query_information()
        while ~(args.information > 0 && args.information <= 1)
            fprintf('How much information would you like to retain?\n');
            args.information = input('Enter Value (Default 0.80): ');
            if isempty(args.information)
                args.information = 0.80;
            end
            if ~isfloat(args.information)
                args.information = -1;
            end
        end
    end

    function query_samples()
        while ~(args.removesamples >= 0)
            fprintf('How many subjects would you like to REMOVE?\n');
            args.removesamples = input('Enter Value (Default 0): ');
            if isempty(args.removesamples)
                args.removesamples = 0;
            end
            if ~isfloat(args.removesamples)
                args.removesamples = -1;
            end
            args.removesamples = round(args.removesamples);
        end
    end

    function query_training()
        % attempt to automatically find the most recent training file with the correct resolution
        if strcmp(args.trainingfile,'auto')
            d = dir(args.resultsdir);
            dir_names = {d([d.isdir]).name};
            
            % Find training data of the same type
            valid_idx = [];
            valid_times = [];
            for dir_idx = 1:length(dir_names)
                [set, count] = sscanf(dir_names{dir_idx}, 'training_%c_*');
                rm_count = Inf;
                if count
                    rm_count = sscanf(dir_names{dir_idx}, ['training_' set '_%d_*']);
                    if isempty(rm_count)
                        rm_count = -1;
                    end
                end
                if count && strcmp(set, testset_str{args.testset}) && rm_count == args.removesamples
                    timestamp = sscanf(dir_names{dir_idx}, ['training_' set '_' int2str(rm_count) '_%s']);
                    valid_idx(end+1) = dir_idx;
                    valid_times(end+1) = datenum(timestamp, 'yyyymmdd_HHMMSSFFF');
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

    % Query for data directory and results directory if defaults not used
    while isempty(args.datadir) || ~isdir(args.datadir)
        args.datadir = uigetdir('.', 'Data Directory');
    end
    while isempty(args.resultsdir) || exist(args.resultsdir, 'file')==2
        args.resultsdir = uigetdir('.', 'Results Directory');
    end
    if ~exist(args.resultsdir, 'dir')
        mkdir(args.resultsdir);
    end

    datestamp = datestr(now,'yyyymmdd_HHMMSSFFF');
    switch args.experiment
        case 0
            query_dataset();
            query_samples();
            argstring = sprintf('training_%s_%d_%s', testset_str{args.testset}, ...
                args.removesamples, datestamp);
            args.resultsdir = [args.resultsdir filesep argstring];
            args.trainingfile = [args.resultsdir filesep 'training.mat'];
            mkdir(args.resultsdir);
            fprintf(1, 'Training results be written to directory %s\n', args.resultsdir);
            fprintf(1, 'Training filename will be %s\n', args.trainingfile);
        case 1
            args.removesamples = 0;
            query_dataset();
            query_information();
            query_training();
            argstring = sprintf('experimentA_%s%03.0f_%s', testset_str{args.testset}, ...
                                round(args.information*100), datestamp);
            args.resultsdir = [args.resultsdir filesep argstring];
            mkdir(args.resultsdir);
            fprintf(1, 'Results will be written to directory %s\n', args.resultsdir);
        case 2
            query_dataset();
            query_samples();
            query_information();
            query_training();
            argstring = sprintf('experimentB_%s%03.0f_%d_%s', testset_str{args.testset}, ...
                                round(args.information*100), args.removesamples, datestamp);
            args.resultsdir = [args.resultsdir filesep argstring];
            mkdir(args.resultsdir);
            fprintf(1, 'Results will be written to directory %s\n', args.resultsdir);
        otherwise
            fprintf('ERROR: Unknown experiment selection\n');
    end
    args.traindir = [args.datadir filesep 'fa_' testset_str{args.testset}];
    args.testdir = [args.datadir filesep 'fb_' testset_str{args.testset}];

    if exist(args.traindir, 'dir') ~= 7
        error('%s does not exist or is not a directory', args.traindir);
    end
    if exist(args.testdir, 'dir') ~= 7
        error('%s does not exist or is not a directory', args.testdir);
    end
end

