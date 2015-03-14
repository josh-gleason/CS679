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

close all; clear all; clc; fclose('all');

%-----------------------------------------------------------
% generate or read feature data
%
% structure for class statistics
nFeats = 2; nClasses = 2;
cp = struct('pC',NaN,'mV',NaN,'cM',NaN,'nS',NaN,'fV',NaN);
cDef = repmat(cp,2,1);

% Class 1 description
% 
cDef(1).mV = [0; 0];
cDef(1).cM = [1  0; 0 1];
cDef(1).pC = 0.5;
cDef(1).nS = 10000;
cDef(1).fV = zeros(nFeats,cDef(1).nS);

% Class 2 description
% 
cDef(2).mV = [0; 0];
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
% In future, use the function call 
%
%        features2keep = selectFeatures(features);
% 
% For now, keep all features or manually select the desired features.
%      
%          features2keep = 1:nFeats;
%
features2keep = 1:nFeats;
% Now keep the desired features.
features = features(features2keep,:);

%-----------------------------------------------------------
% Now train and classify the samples
nSkip = 1000;
fprintf(1,'-------------\n');
for iTrial = 1:5
    
    nKeep = round(cDef(iClass).nS / nSkip);
    
    samples2keep = round(cDef(iClass).nS*rand(1,nKeep));
    
    %-----------------------------------------------------------
    % trainClassifier
    %
    %  Determine the mean and covariance of each class
    %
    
    classScore = zeros(nClasses,nTotalSamples);
    for iClass = 1:nClasses
        
        % Retrieve the feature vector
        fV = cDef(iClass).fV(features2keep,samples2keep);
        
        % Retrieve the a-priori probability
        pC = cDef(iClass).pC;
        
        % Estimate the classifier params
        [meanV, covM] = trainClassifier(fV);
%         fprintf(1,'class = %d\n',iClass);
%         disp(meanV);
%         disp(covM);
        
        % Classifiy all samples according to the classifier
        classScore(iClass,:) = bayesianClassifier(features, meanV, covM, pC);
        
        figure; plot(1:nTotalSamples,classScore(iClass,:),'.');
        
        title(sprintf('mapping all samples class %d',iClass));
        grid on; xlabel('sample'); ylabel('classifier mapping');
        
    end
    [score, decisions]=max(classScore,[],1);
    
    fprintf(1,'Classifier results:\n');
    truePositives  = zeros(nClasses,1);
    falsePositives = zeros(nClasses,1);
    falseNegatives = zeros(nClasses,1);
    for iClass = 1:nClasses
        
        truthSameClass  = (truth == iClass);
        truthDiffClass  = (truth ~= iClass);
        
        nSS = cDef(iClass).nS;       % number of samples in same class
        nSD = nTotalSamples - nSS;   % number of samples in diff class
        
        truePositives(iClass)  = sum(decisions(truthSameClass)==iClass)/nSS;
        falsePositives(iClass) = sum(decisions(truthDiffClass)==iClass)/nSD;
        falseNegatives(iClass) = sum(decisions(truthSameClass)~=iClass)/nSS;
        
        fprintf(1,'\tiClass = %d tp = %f fp = %f fn = %f\n',iClass,...
            truePositives(iClass),falsePositives(iClass),falseNegatives(iClass));
    end
    
end


