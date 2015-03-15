%-----------------------------------------------------------
%
%  Program: testingClassifierFaceDetector
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

[fnParams, pnParams] = uigetfile('*.dat','select classifier parameters');

classParams = dlmread([pnParams filesep fnParams]);

[nClasses, nColumns] = size(classParams);
nFeatures = classParams(1,2);

nc = input('Number of classifier params to use? 1, 2, or 3 ','s');
switch lower(nc)
    case '1'
       classes2test=1;
    case '2'
       classes2test=2;
    case '3'
       classes2test=3;
    otherwise
        error('reenter number of classes');
end

%----------------------------------------------------------
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
    
    classScore = zeros(numel(classes2test),nTotalSamples);
        
    for iClass = classes2test
        
        % Classifier params
        pC = classParams(iClass,1);
        
        meanV = classParams(iClass,3:3+nFeatures-1)';
        covM  = reshape(classParams(iClass,nFeatures+3:end),nFeatures,nFeatures);
        
        %         fprintf(1,'class = %d\n',iClass);
        %         disp(meanV);
        %         disp(covM);
        
        % Classifiy all samples according to the classifier
        classScore(iClass,:) = bayesianClassifier(features, meanV, covM, pC);
        
        figure;
        plotData =[nan(1,cDef(1).nS) classScore(iClass,cDef(1).nS+1:end)];
        plot(1:nTotalSamples,plotData(1:end),'r.');
        hold on;
        plot(1:cDef(1).nS,classScore(1,1:cDef(1).nS),'b.');
        title(sprintf('ln P(w=%d|f)',iClass));
        legend('background','face');
        hold off;
        
        figure(180); hist(classScore(iClass,:),50);
        
        title(sprintf('histogram of scores for all samples class %d',iClass));
        grid on; xlabel('sample'); ylabel('classifier score');
        
    end
    %[scoreAll, decisions]=max(classScore,[],1);
    
%     fprintf(1,'Classifier results:\n');
%     truePositives  = zeros(nClasses,1);
%     falsePositives = zeros(nClasses,1);
%     falseNegatives = zeros(nClasses,1);
%     for iClass = classes2test
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

% plot the results
truthSameClass  = truthData == 1;
truthDiffClass  = truthData ~= 1;

dataSame = classScore(1,truthSameClass);
dataDiff = classScore(1,truthDiffClass);

minScore = min(min(dataSame(:)),min(dataDiff(:)));
maxScore = max(max(dataSame(:),max(dataDiff(:))));
[histScores, histIndices]= hist([dataSame dataDiff],minScore:maxScore);
pdfScores = histScores/sum(histScores(:));
cdfScores = cumsum(pdfScores);

minIndex = find(cdfScores < 0.01);
minIndex = minIndex(end);
maxIndex = find(cdfScores > 0.9999);
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


