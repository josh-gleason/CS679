%------------------------------------------------------------
%
%  Program: trainClassifier
%
%  Purpose: estimate the parameters to the Gaussian model for the
%  classifier using maximum likelihood methods.
%
%
%  Input: 
%
%     dataMatrix: nFeatures x nSamples
%          f11   f12   f12   f14 ...
%          f21   f22   f23   f24 ...
%  
%          fn1   fn2   fn3   fn4 ...
%
%     nColSkip:  number samples to skip in training
%         e.g. nColSkip = 5, keep 1 out of 5
%
%     featuresKeep: array of features to keep
%         e.g. featuresKeep = [3 6 8]
%
%  Output:
%
%     meanV: n x 1  (mean vector of features for class)
%
%     covM: n x n    (covariance matrix of features for class)
%
%  Programmer: Rod Pickens
%
%  Date: March 13, 2015
%
%-------------------------------------------------------------

function [meanV, covM] = trainClassifier(dataMatrix)
%clc; close all; clear all; fclose('all');

% % Input parameters
% nSkip = 1; 
% featuresKeep = [1 2];
% 
% [fn, pn]=uigetfile('*.dat','input training data');
% po = uigetdir(pn,'path for parameter file');
% 
% % Read the data
% dataMatrix = csvread([pn filesep fn]);

% Keep only the features selected
[nr, ~] = size(dataMatrix);

% Now train the classifier using maximum likelihood assuming
% a model of Gaussian
if nr == 1
   meanV = mean(dataMatrix);
   covM  = cov(dataMatrix);
else
   meanV = mean(dataMatrix,2);
   covM  = cov(dataMatrix,dataMatrix');
end

% Save the training for the classifier
%fileOut = sprintf('%s_nSamples_%d',strrep(fn,'data.dat','params.dat'),nKept);

%csvwrite([po filesep fileOut],[meanV(:)' covM(:)']);

