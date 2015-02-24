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
function project1()
    p1.mu1    = [1;1];
    p1.sigma1 = eye(2);
    p1.mu2    = [4;4];
    p1.sigma2 = eye(2);
    g1        = @discriminant1;
    b1        = @boundry1;
    
    p2.mu1    = [1;1];
    p2.sigma1 = eye(2);
    p2.mu2    = [4;4];
    p2.sigma2 = [4 0; 0 16];
    g2        = @discriminant2;
    b2        = @boundry2;

    close('all');
    run_experiment(p1, g1, b1, 1);
    run_experiment(p2, g2, b2, 2);
end
