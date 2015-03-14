%------------------------------------------------------------
%
%  Program: estimateClassifierParameters
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

function [meanV, covM] = estimateClassifierParameters(dataMatrix)
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
[nr, nc] = size(dataMatrix);

% Now train the classifier using maximum likelihood assuming
% a model of Gaussian
if nr == 1
   meanV = mean(dataMatrix);
   covM  = cov(dataMatrix);
else
   meanV = mean(dataMatrix,2);
   covM  = 1/(nc-1)*(dataMatrix*dataMatrix') - meanV * meanV';
end

% Save the training for the classifier
%fileOut = sprintf('%s_nSamples_%d',strrep(fn,'data.dat','params.dat'),nKept);

%csvwrite([po filesep fileOut],[meanV(:)' covM(:)']);

