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
pnFigures = uigetdir('.','save figures to which directory');

[fnFeats, pnFeats] = uigetfile('*.mat','select feature file');
[~, fileName, ~]=fileparts(fnFeats);
    
load([pnFeats filesep fnFeats]);

pnParams = uigetdir('.','select directory to place classifier parameters');

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

dataSame = classScore(1,truthSameClass);
dataDiff = classScore(1,truthDiffClass);

minScore = min(classScore(:));
maxScore = max(classScore(:));
[histScores, histIndices]= hist(classScore(1,:),minScore:maxScore);
pdfScores = histScores/sum(histScores(:));
cdfScores = cumsum(pdfScores);

minIndex = find(cdfScores < 0.01);
minIndex = minIndex(end);
maxIndex = find(cdfScores > 0.99);
maxIndex = maxIndex(1);
pdfDomain = histIndices(minIndex):0.1:histIndices(maxIndex);  % a reasonable range for the feature scores

histSame = hist(dataSame,pdfDomain); sumSame=sum(histSame);
histDiff = hist(dataDiff,pdfDomain); sumDiff=sum(histDiff);

pdfSame = histSame/sumSame;
pdfDiff = histDiff/sumDiff;

cdfSame = cumsum(pdfSame);
cdfDiff = cumsum(pdfDiff);

figure(250); 
plot(pdfDomain,cdfSame,'b',pdfDomain,cdfDiff,'r-');
title('cumulative density functions for both classes');
xlabel('classifier score'); ylabel('prob');
legend('same','diff'); grid on;
saveas(250,[pnFigures filesep sprintf('cdf_features')],'bmp');

figure(350); 
plot(cdfDiff(1:end),cdfSame(end:-1:1),'b');
axis([0 1 0 1]); grid on;
xlabel('Pmiss=Pfn: false negative rate (FNR)');
ylabel('Pfa=Pfp: false positive rate (FPR)');
title('ROC comparing FNR versus FPR');
saveas(350,[pnFigures filesep sprintf('FPRversusFNR')],'bmp');

figure(450); 
plot(1-cdfDiff(end:-1:1),1-cdfSame(end:-1:1),'b');
axis([0 1 0 1]); grid on;
xlabel('Pfa: false positive rate (FPR)');
ylabel('Pd: true positive rate (TPR)');
title('ROC comparing TPR versus FPR');
saveas(450,[pnFigures filesep sprintf('TPRversusFPR')],'bmp');

