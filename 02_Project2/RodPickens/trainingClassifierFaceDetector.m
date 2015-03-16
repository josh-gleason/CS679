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
load('TrainFolders');
%pnFigures = uigetdir('.','save figures to which directory');

%[fnFeats, pnFeats] = uigetfile('*.mat','select feature file');
[~, fileName, ~]=fileparts(fnFeats);
    
load([pnFeats filesep fnFeats]);

%pnParams = uigetdir('.','select directory to place classifier parameters');

fid = fopen([pnParams filesep fileName '.dat'],'w');

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
cV = {'ro','b.'};
figure(100);
hold on;

for iClass = nClasses:-1:1
    
    fV = cDef(iClass).fV(features2keep,:);
    
    plot(fV(1,:),fV(2,:),cV{iClass});
    xlabel('feature 1'); ylabel('feature 2');
    title('face/other class separability');
    grid on;
    
end
saveas(100,[pnFigures filesep 'classSeparability'],'bmp');
%--------------------------------------------------------
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

        fprintf(fid,'%f',pC);
        fprintf(fid,',%d',numel(features2keep));
        fprintf(fid,',%f',meanV);
        fprintf(fid,',%f',covM);
        if iClass < nClasses, fprintf(fid,'\n'); end
        
%         fprintf(1,'class = %d\n',iClass);
%         disp(meanV);
%         disp(covM);
        
        % Classifiy all samples according to the classifier
        classScore(iClass,:) = bayesianClassifier(features, meanV, covM, pC);
        
        figure(80); 
        plotData =[nan(1,cDef(1).nS) classScore(iClass,cDef(1).nS+1:end)];
        plot(1:nTotalSamples,plotData(1:end),'r.'); 
        hold on;
        plot(1:cDef(1).nS,classScore(1,1:cDef(1).nS),'b.'); 
        title(sprintf('ln P(w=%d|f)',iClass));
        legend('background','face'); grid on;
        hold off;
        saveas(80,[pnFigures filesep sprintf('classifierScores_class=%d',iClass)],'bmp');

        figure(180); hist(classScore(iClass,:),50);
        title(sprintf('histogram of scores for all samples class %d',iClass));
        grid on; xlabel('sample'); ylabel('classifier score');
        saveas(180,[pnFigures filesep sprintf('classifierScoresHist_class=%d',iClass)],'bmp');
       
    end
%     [scoreAll, decisions]=max(classScore,[],1);
%     
%     fprintf(1,'Classifier results:\n');
%     truePositives  = zeros(nClasses,1);
%     falsePositives = zeros(nClasses,1);
%     falseNegatives = zeros(nClasses,1);
%     for iClass = 1:nClasses
%         
%         truthSameClass  = (truthData == iClass);
%         truthDiffClass  = (truthData ~= iClass);
%         
%         nSS = cDef(iClass).nS;       % number of samples in same class
%         nSD = nTotalSamples - nSS;   % number of samples in diff class
%         
%         truePositives(iClass)  = sum(decisions(truthSameClass)==iClass)/nSS;
%         falsePositives(iClass) = sum(decisions(truthDiffClass)==iClass)/nSD;
%         falseNegatives(iClass) = sum(decisions(truthSameClass)~=iClass)/nSS;
%         
%         fprintf(1,'\tiClass = %d tp = %f fp = %f fn = %f\n',iClass,...
%             truePositives(iClass),falsePositives(iClass),falseNegatives(iClass));
%         
%     end
    
end
fclose(fid);

% plot the results
truthSameClass  = truthData == 1;
truthDiffClass  = truthData ~= 1;

scoreDiff = classScore(1,:) - classScore(2:end,:);
dataSame = scoreDiff(truthSameClass);
dataDiff = scoreDiff(truthDiffClass);

% Show class scores as image
for iClass = 1:nClasses
    figure(120+4*(iClass-1));
    [r,c,~] = size(trainImg);
    imageScore = zeros(r,c);
    imageScore(fIdx) = classScore(iClass,truthSameClass);
    imageScore(bIdx) = classScore(iClass,truthDiffClass);
    imagesc(exp(imageScore));
    colorbar;
    title(sprintf('Discriminate Score Class (log normalized) %d', iClass));

    % plot classification results
    figure(121+4*(iClass-1));
    [~,idx] = max(classScore);
    decisionIdx = (idx==iClass);
    imageClass = zeros(r,c);
    imageClass(fIdx) = decisionIdx(truthSameClass);
    imageClass(bIdx) = decisionIdx(truthDiffClass);
    imagesc(imageClass);
    title(sprintf('Class %d classification results (Yellow for classified)', iClass));
    
    if iClass == 1
        figure(122+4*(iClass-1));
        imageTruth = zeros(r,c);
        imageTruth(fIdx) = truthSameClass(truthSameClass);
        imagesc(imageTruth);
        title(sprintf('Truth Class %d', iClass));

        figure(123+4*(iClass-1));
        imageRes = zeros(r,c);
        % 0: false negative
        % 1: false positive
        % 2: true negative
        % 3: true positive
        imageRes(imageTruth ~= imageClass & imageClass == 0) = 0;
        imageRes(imageTruth ~= imageClass & imageClass == 1) = 1;
        imageRes(imageTruth == imageClass & imageClass == 0) = 2;
        imageRes(imageTruth == imageClass & imageClass == 1) = 3;
        imagesc(imageRes);
        colorbar;
        title(sprintf('0: False Negative\n1: False Positive\n2: True Negative\n3: True Position'));
%         FN = sum(imageRes(:)==0)
%         FP = sum(imageRes(:)==1)
%         TN = sum(imageRes(:)==2)
%         TP = sum(imageRes(:)==3)
%         
%         TPR = TP/(TP+FN)
%         FPR = FP/(FP+TN)
%         FNR = FN/(FN+TP)
    end
end

minScore = min(scoreDiff);
maxScore = max(scoreDiff);
[histScores, histIndices] = hist(scoreDiff,linspace(minScore,maxScore,1000));
pdfScores = histScores/sum(histScores(:));
cdfScores = cumsum(pdfScores);

minIndex = find(cdfScores < 0.01, 1, 'last');
if isempty(minIndex), minIndex = 1; end
maxIndex = find(cdfScores > 0.99, 1, 'first');
if isempty(maxIndex), maxIndex = 1000; end
pdfDomain = linspace(histIndices(minIndex),histIndices(maxIndex),1000);

histSame = hist(dataSame,pdfDomain); sumSame=sum(histSame);
histDiff = hist(dataDiff,pdfDomain); sumDiff=sum(histDiff);

pdfSame = histSame / sumSame;
pdfDiff = histDiff / sumDiff;

cdfSame = cumsum(pdfSame);
cdfDiff = cumsum(pdfDiff);

figure(250); 
plot(pdfDomain,cdfSame,'b',pdfDomain,cdfDiff,'r-');
title('cumulative density functions for both classes');
xlabel('classifier score'); ylabel('prob');
legend('same','diff'); grid on;
saveas(250,[pnFigures filesep sprintf('cdf_features')],'bmp');

figure(350); 
plot(1-cdfDiff(1:end),cdfSame(1:end),'b');
axis([0 1 0 1]); grid on;
xlabel('Pfa=Pfp: false positive rate (FPR)');
ylabel('Pmiss=Pfn: false negative rate (FNR)');
title('ROC comparing FNR versus FPR');
saveas(350,[pnFigures filesep sprintf('FPRversusFNR')],'bmp');

figure(450); 
plot(1-cdfDiff(end:-1:1),1-cdfSame(end:-1:1),'b');
axis([0 1 0 1]); grid on;
xlabel('Pfa: false positive rate (FPR)');
ylabel('Pd: true positive rate (TPR)');
title('ROC comparing TPR versus FPR');
saveas(450,[pnFigures filesep sprintf('TPRversusFPR')],'bmp');

