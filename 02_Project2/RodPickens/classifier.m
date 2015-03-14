%-----------------------------------------------------------
%
%  Program: classifier
%
%  Purpose: Classify samples into one or more classes
%
%  Programmer: Rod Pickens
%
%  Date: March 13, 2015
%
%-----------------------------------------------------------

%-----------------------------------------------------------
% generate or read feature data
%
% structure for class statistics
nFeats = 2; nClasses = 2;
cp = struct('pC',NaN,'mV',NaN,'cM',NaN,'nS',NaN,'fV',NaN);
cDef = repmat(cp,2,1);

% Class 1 description
% 
cDef(1).mV = [1; 1];
cDef(1).cM = [1  0; 0 1];
cDef(1).pC = 0.5;
cDef(1).nS = 10000;
cDef(1).fV = zeros(nFeats,cDef(1).nS);

% Class 2 description
% 
cDef(2).mV = [4; 4];
cDef(2).cM = [1  0; 0  1];
cDef(2).pC = 1 - cDef(1).pC;
cDef(2).nS = 10000;
cDef(2).fV = zeros(nFeats,cDef(2).nS);

features = []; truth = [];
nTotalSamples = 0;
for iClass = 1:nClasses
   % Generate the synthetic feature sets
   nSamples = cDef(iClass).nS; 
   meanV    = cDef(iClass).mV;
   covM     = cDef(iClass).cM;
   cDef(iClass).fV = generateSyntheticData(nSamples,meanV,covM);
   features = [features cDef(iClass).fV];
   truth    = [truth repmat(iClass,1,nSamples)];
   nTotalSamples = nTotalSamples + nSamples;
end
%-----------------------------------------------------------
% selectFeatures: 
%
%    chose a subset of the n features for classification.
%
% In future, use the function call features2keep = selectFeatures(features);
% 
% For now, keep all features.
features2keep = 1:nFeats;

%-----------------------------------------------------------
% trainClassifier
% 
%  Determine the mean and covariance of each class
%  
nSkip = 1;
cp = struct('bd',NaN);
classDec  = repmat(cp,nClasses,1);
decisions = -inf(1,nTotalSamples);
for iClass = 1:nClasses
   
   % Retrieve the feature vector
   fV = cDef(iClass).fV; 
   
   % Retrieve the a-priori probability
   pC = cDef(iClass).pC;
   
   % Estimate the classifier params
   [meanV, covM] = trainClassifier(fV, nSkip, features2keep);
   
   % Classifiy all samples according to the classifier
   classDec(iClass).bd = zeros(1,nTotalSamples);
   classDec(iClass).bd = bayesianClassifier(features, meanV, covM, pC);
   
   % Determine if new probs are 
   selection = classDec(iClass).bd > decisions ;
   decisions(selection) = iClass;
   
end

truePositives  = zeros(nClasses,1);
falsePositives = zeros(nClasses,1);
falseNegatives  = zeros(nClasses,1);
for iClass = 1:nClasses
  sameClass = decisions(decisions == iClass);
  diffClass = decisions(decisions ~= iClass);
   
  truePositives(iClass)  = decisions(sameClass) == truth(truth == iClass);
  falsePositives(iClass) = decisions(sameClass) == truth(truth ~= iClass);
  falseNegatives(iClass) = decisions(diffClass) == truth(truth == iClass); 

end



