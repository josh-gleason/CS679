function main(varargin)
    args = get_args(varargin);
    
    close('all');
    switch args.experiment
        case 0
            %training(args);
        case 1
            %experiment_1(args);
        case 2
            %experiment_2(args);
    end
end