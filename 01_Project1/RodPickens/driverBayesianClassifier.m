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
[fileIn, pathIn]=uigetfile({'*.txt'},'classifier file');

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
for iClass = 1:nClasses
    classifierParams(iClass).pClass = classParams(iClass,1);
    classifierParams(iClass).meanV  = classParams(iClass,3:3+nFeatures-1)';
    classifierParams(iClass).covM   = reshape(classParams(iClass,nFeatures+3:end),nFeatures,nFeatures);
end
%---------------------------------------------------------------
% B) Create nSamples for each class with each sample
%    containing nFeatures
%---------
%
%   sample 1 = x11, x12, x13, ..., x1n
%   sample 2 = x21, x22, x23, ..., x2n
% etc.

% Currently, I don't like this structure.  Rethink.
nSamplesPerClass = 10000;
s = struct('features',NaN(nFeatures,nSamplesPerClass),...
           'classifierDecision',NaN(1,nSamplesPerClass),...
           'minFeatureValue',NaN(nFeatures),...
           'maxFeatureValue',NaN(nFeatures),...
           'decisionStatus',NaN(1,nSamplesPerClass));
       
samples = repmat(s,nClasses,nSamplesPerClass);

% Create n samples for each class with each sample being an n-Dimensional
% vector of features.
%
%     x(1:nSamples,1:nFeatures) = N(u, S) 
%
minV = inf;
maxV = -inf;
for iClass = 1: nClasses
    samples(iClass).features = createClasses(nSamplesPerClass,classifierParams(iClass));
    
    samples(iClass).minFeatureValue = min([samples(iClass).features(1,:) ...
                                           samples(iClass).features(2,:)]);
    
    samples(iClass).maxFeatureValue = max([samples(iClass).features(1,:) ...
                                           samples(iClass).features(2,:)]);
end 
figure(100); 
plot(samples(1).features(1,:),samples(1).features(2,:),'b.');
hold on; grid on;
plot(samples(2).features(1,:),samples(2).features(2,:),'r.');

%--------------------------------------------------------------------
% Create a meshgrid plot of the Gaussian distributions
% to visually determine if the random numbers look Gaussian.
minF1 = -5; maxF1 = 10;
minF2 = -5; maxF2 = 10;
deltaF1 = max(ceil(maxF1 - minF1),ceil(maxF2 - minF2));
deltaF2 = deltaF1;
df1 = deltaF1 / 500;
df2 = deltaF2 / 500;
[f1, f2]=meshgrid(minF1:df1:maxF1,minF2:df2:maxF2);

pxW1 = normalDensity([f1(:)'; f2(:)'],classifierParams(1));
pxW2 = normalDensity([f1(:)'; f2(:)'],classifierParams(2));

figure; 
mesh(f1,f2,reshape(pxW1,size(f1))); hold on;
mesh(f1,f2,reshape(pxW2,size(f2)));

%--------------------------------------------------------------------
%
%  Determine the probability of error and the Chernoff and
%  Bhattacharyya error bounds.
%
bFactor = 0:0.05:1;
pW1 = classifierParams(1).pClass;
pW2 = classifierParams(2).pClass;
pError = zeros(length(bFactor),1);
chernoffErr = zeros(length(bFactor),1);
for iB = 1:numel(bFactor)
    bV = bFactor(iB);
    pError(iB) = (pW1^bV)*(pW2^(1-bV))*df1*df2*sum(sum((pxW1.^(bV)).*(pxW2.^(1-bV))));
    chernoffErr(iB) = chernoffErrorNormal(bV, classifierParams);
end
bhattacharyaaErr = bhattacharyaaErrorNormal(classifierParams);
fprintf(1,'P(error)=%f Chernoff bound=%f  Bhattcharyya bound=%f\n',...
    min(pError(:)),min(chernoffErr(:)),min(bhattacharyaaErr(:)));

% Plot prob of error for varying beta.
figure; 
plot(bFactor,pError,'b');
title('Probability of Error');
axis([0 1.1 0 1.1]); grid on;
xlabel('\beta'); ylabel('P(error)');

%----------------------------------------------------------------
% C) Now classify all the samples from all the classes
%-----
%
for iClass = 1:nClasses
  classifierScore = zeros(nClasses,nSamplesPerClass);
  for iClassifier = 1:nClasses
    classifierScore(iClassifier,:) = bayesianClassifier(samples(iClass).features,classifierParams(iClassifier));
  end
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

