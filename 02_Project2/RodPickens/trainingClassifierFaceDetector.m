%-----------------------------------------------------------
%
%  Program: trainingClassifierFaceDetector
%
%  Purpose: Select features, estimate classifier parameters,
%     classify samples into one or more classes, estimate
%     performance using standard statistics.
%
%  Programmer: Rod Pickens
%
%  Date: March 13, 2015
%
%-----------------------------------------------------------

close all; clear variables; clc; fclose('all');

%-----------------------------------------------------------
% generate or read feature data
%
% structure for class statistics

[fnFeats, pnFeats] = uigetfile('*.mat','select feature file');
[fileName, ~]=fileparts(fnFeats);
    
load([pnFeats filesep fnFeats]);

pnParams = uigetdir('.','select directory to place classifier parameters');

fid = fopen([pnParams filesep strrep(fileName,'Data','Params') '.dat'],'w');

%-----------------------------------------------------------
% selectFeatures: 
%
%    Chose a subset of the n features for classification.
%
% In future, use the function call 
%
%        features2keep = selectFeatures(features);
% 
% For now, keep all features or manually select the desired features.
%      
%        features2keep = 1:nFeats;
%
features2keep = 1:nFeats;
% Now keep the desired features.
features = features(features2keep,:);

%-----------------------------------------------------------
% Now train and classify the samples
nSkip = 1;

fprintf(1,'-------------\n');
for iTrial = 1:1
        
    %-----------------------------------------------------------
    % trainClassifier
    %
    %  Determine the mean and covariance of each class
    %
    
    classScore = zeros(nClasses,nTotalSamples);
    
    for iClass = 1:nClasses
        
        if nSkip > 1
           nKeep = round(cDef(iClass).nS / nSkip);
           samples2keep = round(cDef(iClass).nS*rand(1,nKeep));
           samples2keep(samples2keep < 1) = 1;
           samples2keep(samples2keep > cDef(iClass).nS) = cDef(iClass).nS;
        else
           samples2keep = 1:cDef(iClass).nS;
        end
    
        % Retrieve the feature vector
        fV = cDef(iClass).fV(features2keep,samples2keep);
        
        % Retrieve the a-priori probability
        pC = cDef(iClass).pC;
        
        % Estimate the classifier params
        [meanV, covM] = estimateClassifierParameters(fV);

        fprintf(fid,'%d',iClass);
        fprintf(fid,',%f',meanV);
        fprintf(fid,',%f',covM);
        if iClass < nClasses, fprintf(fid,'\n'); end
        
%         fprintf(1,'class = %d\n',iClass);
%         disp(meanV);
%         disp(covM);
        
        % Classifiy all samples according to the classifier
        classScore(iClass,:) = bayesianClassifier(features, meanV, covM, pC);
        
        figure; 
        plot(1:nTotalSamples,[nan(1,cDef(1).nS) classScore(iClass,cDef(1).nS+1:end)],'r.'); hold on;
        plot(1:cDef(1).nS,classScore(1,1:cDef(1).nS),'b.'); hold off;
        title(sprintf('ln P(w=%d|f)',iClass));
        legend('background','face');
        
        figure; hist(classScore(iClass,:),50);
        
        title(sprintf('mapping all samples class %d',iClass));
        grid on; xlabel('sample'); ylabel('classifier mapping');
        
    end
    [score, decisions]=max(classScore,[],1);
    
    fprintf(1,'Classifier results:\n');
    truePositives  = zeros(nClasses,1);
    falsePositives = zeros(nClasses,1);
    falseNegatives = zeros(nClasses,1);
    for iClass = 1:nClasses
        
        truthSameClass  = (truthData == iClass);
        truthDiffClass  = (truthData ~= iClass);
        
        nSS = cDef(iClass).nS;       % number of samples in same class
        nSD = nTotalSamples - nSS;   % number of samples in diff class
        
        truePositives(iClass)  = sum(decisions(truthSameClass)==iClass)/nSS;
        falsePositives(iClass) = sum(decisions(truthDiffClass)==iClass)/nSD;
        falseNegatives(iClass) = sum(decisions(truthSameClass)~=iClass)/nSS;
        
        fprintf(1,'\tiClass = %d tp = %f fp = %f fn = %f\n',iClass,...
            truePositives(iClass),falsePositives(iClass),falseNegatives(iClass));
    end
    
end

% plot the results
minScore = min(classScore(:));
pdfDomain = minScore:0.5:0;  % a reasonable range for the feature scores

for iClass = 1:nClasses
   
   nSS = cDef(iClass).nS;
   nSD = nTotalSamples - nSS;
   
   % prob density function for true positives on face
   pdfTP = hist(classScore(iClass,truthData == iClass),pdfDomain)/nSS;
   cdfTP = cumsum(pdfTP);

   % prob density function for false negatives on face
   pdfFP = hist(classScore(iClass,truthData ~= iClass),pdfDomain)/nSD;
   cdfFP = cumsum(pdfFP);

   figure; plot(pdfDomain,cdfTP,'b',pdfDomain,cdfFP,'r'); 
   title(sprintf('distribution on class P(%d)=%f',iClass,cDef(iClass).pC));
   legend('TP','FP'); grid on;
   figure; plot(cdfFP,cdfTP,'b'); 
   title(sprintf('ROC for class %',iClass));
   xlabel('FP'); ylabel('TP'); grid on;
   
%    % plot the results
%    pdfFPbckgnd = hist(bckgndClassifierScore(1,:),pdfDomain)/nSamplesInBckgndClass;
%    cdfFPbckgnd = cumsum(pdfFPbckgnd);
%    pdfTPbckgnd = hist(bckgndClassifierScore(2,:),pdfDomain)/nSamplesInBckgndClass;
%    cdfTPbckgnd = cumsum(pdfTPbckgnd);
% 
%    figure; plot(pdfDomain,pdfTPbckgnd,'b',pdfDomain,pdfFPbckgnd,'r'); title('background');
end

fclose(fid);


