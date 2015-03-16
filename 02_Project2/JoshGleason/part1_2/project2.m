%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% CS 679 (Pattern Recognition)
% Spring 2015
% Programming Assignment 1
%
% Author: Josh Gleason
%
% Date: 02/10/2015
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function project2()
    p1.mu1    = [1;1];
    p1.sigma1 = eye(2);
    p1.mu2    = [4;4];
    p1.sigma2 = eye(2);
    
    p2.mu1    = [1;1];
    p2.sigma1 = eye(2);
    p2.mu2    = [4;4];
    p2.sigma2 = [4 0; 0 16];

    close('all');
    
    exp = questdlg('Which Experiment would you like to run?', 'Experiment', 'Standard', 'Error Rate', 'ER vs N Training', 'Standard');
    
    if strcmp(exp, 'Standard')
        experiment = @run_experiment;
    elseif strcmp(exp, 'Error Rate')
        experiment = @run_experiment_1000;
    elseif strcmp(exp, 'ER vs N Training')
        experiment = @run_experiments;
    else
        return;
    end
    
    experiment(p1, 1);
    experiment(p2, 2);
end
