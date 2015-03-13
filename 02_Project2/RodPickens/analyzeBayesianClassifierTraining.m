%------------------------------------------------
%
%   Program: driverBayesianClassifier
%
%   Purpose: design a Bayesian classifier to classify
%   samples from n classes with n mean vectors, n covariance
%   matrices, and n a-priori probabilities.  Each 
%   class will have decision based on m features.
%
%   Math approach:
%
%   Decide class membership by finding 
%
%       wi = arg max (g1(x), g2(x), ..., gi(x), ... gn(x)
%
%   where
%
%       gi(x) = ln( p(x|wi) ) + p(wi)    i = 1:n
%
%   and where p(x|wi) is a gaussian PDF with distribution
%
%       N(meanWi, sigmaWi)
%
%   with mean vector meanWi and covariance matrix sigmaWi
%   and with p(wi) being the apriori probability for class wi.
%
%   Programmer: Rod Pickens
%
%   Date: Feb 10, 2015
%
%--------------------------------------------------

clear all; close all; clc;  fclose('all');

%--------------------------------------------------
% A) Input classifier parameters for n classes
%---------
% Select classifier input file
[fileIn, pathIn]=uigetfile({'*.*'},'classifier parameter file');

fprintf(1,'-----\nInput file = %s\n-----\n',fileIn);
% Read the classifier parameters file
classParams = dlmread([pathIn filesep fileIn]);

% Determine size of file
% 
%   Format of file:
%
%   row 1 has class 1 mean vector u(:), covariance matrix S(:)
%   row 2 has class 2 mean vector u(:), covariance matrix S(:)
%   
%   row n has class n mean vector u(:), covariance matrix S(:)
%
[nClasses, nColumns] = size(classParams);
nFeatures = classParams(1,2);

cp = struct('pClass',NaN,...
    'meanV',NaN(nFeatures,1),...
    'covM',NaN(nFeatures,nFeatures));

classifierParams = repmat(cp,nClasses,1);

maxNumberSamples = max(nSamplesPerClass);
s = struct('features',NaN(nFeatures,maxNumberSamples),...
           'classifierDecision',NaN(1,maxNumberSamples),...
           'decisionStatus',NaN(1,maxNumberSamples));

nSamplesPerClass = [10000; 10000];
for iClass = 1:nClasses
   
    nSamples = nSamplesPerClass(iClass);
    
    % Classifier params
    classifierParams(iClass).pClass = classParams(iClass,1);
    classifierParams(iClass).meanV  = classParams(iClass,3:3+nFeatures-1)';
    classifierParams(iClass).covM   = reshape(classParams(iClass,nFeatures+3:end),nFeatures,nFeatures);

    % Read the class features
    samples = repmat(s,1,nSamplesPerClass);
    samples.features = createClasses(nSamplesPerClass,classifierParams(iClass));
    
    errorBounds(classifierParams)

%----------------------------------------------------------------
% C) Now classify all the samples from all the classes
%-----
%
    classifierScore = zeros(1,nSamplesPerClass);

    classifierScore(:) = bayesianClassifier(samples(iClass).features,classifierParams(iClassifier));

    [maxValue, maxIndex]=max(classifierScore,[],1);
    samples(iClass).classifierDecision = maxIndex;
    Pd = sum(maxIndex==iClass)/numel(maxIndex);
    Pfa = sum(maxIndex ~= iClass & maxIndex ~= 0)/numel(maxIndex);
  fprintf(1,'iClass=%d Pd = %f Pfa = %f\n',iClass,Pd, Pfa);
end

% Now plot and calculate actual error rates.
indexCorrect = find(samples(1).classifierDecision == 1);
figure(200); 
plot(samples(1).features(1,indexCorrect),samples(1).features(2,indexCorrect),'b.');
hold on; grid on;

indexCorrect = find(samples(2).classifierDecision == 2);
plot(samples(2).features(1,indexCorrect),samples(2).features(2,indexCorrect),'r.');
hold on; grid on;

indexInCorrect = find(samples(2).classifierDecision == 1);
plot(samples(2).features(1,indexInCorrect),samples(2).features(2,indexInCorrect),'ro','LineWidth',2);

indexInCorrect = find(samples(1).classifierDecision == 2);
plot(samples(1).features(1,indexInCorrect),samples(1).features(2,indexInCorrect),'bo','LineWidth',2);

legend('correct class 1', 'error class 2', 'correct class 2', 'error class 1');

