%------------------------------------------------------------
%
%  Program: trainClassifier
%
%  Purpose: estimate the parameters to the model for the
%  classifier using maximum likelihood methods
%
%  Programmer: Rod Pickens
%
%  Date: March 13, 2015
%
%-------------------------------------------------------------

clc; close all; clear all; fclose('all');

% Input parameters
nSkip = 1; 
featuresKeep = [1 2];

[fn pn]=uigetfile('*.dat','input training data');
po = uigetdir(pn,'path for parameter file');

% Read the data
dataMatrix = csvread([pn filesep fn]);

% Keep only the features selected
[nr, nc] = size(dataMatrix);
colKeep = 1:nSkip:nc;
nKept   = numel(colKeep);

dataMatrix = dataMatrix(featuresKeep,colKeep);

% Now train the classifier using maximum likelihood assuming
% a model of Gaussian
meanV = mean(dataMatrix(:,colKeep),1);
covM  = cov(dataMatrix(:,colKeep));

% Save the training for the classifier
fileOut = sprintf('%s_nSamples_%d',strrep(fn,'data.dat','params.dat'),nKept);

csvwrite([po filesep fileOut],[meanV(:)' covM(:)']);

