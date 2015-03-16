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

classes2test = input(sprintf('Number of classes to use? max = %d ',nClasses));

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
    
    classScore = zeros(classes2test,nTotalSamples);
        
    for iClass = 1:classes2test
        
        % Classifier params
        pC = classParams(iClass,1);
        
        meanV = classParams(iClass,3:3+nFeatures-1)';
        covM  = reshape(classParams(iClass,nFeatures+3:end),nFeatures,nFeatures);
        
        fprintf(1,'class = %d\n',iClass);
        disp(meanV);
        disp(covM);
        
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

scoreDiff = classScore(1,:) - max(classScore(2:end, :), [], 1);
dataSame = scoreDiff(1,truthSameClass);
dataDiff = scoreDiff(1,truthDiffClass);

% Show class scores as image
for iClass = 1:nClasses
    figNum = 120+4*(iClass-1);
    figure(figNum);
    r = imgRows;
    c = imgCols;
    imageScore = zeros(r,c);
    imageScore(fIdx) = classScore(iClass,truthSameClass);
    imageScore(bIdx) = classScore(iClass,truthDiffClass);
    imagesc(exp(imageScore));
    colorbar;
    title(sprintf('Discriminate Score Class (log normalized) %d', iClass));
    saveas(figNum, [pnFigures filesep sprintf('score_image_%d', iClass)], 'bmp');

    % plot classification results
    figNum = 121+4*(iClass-1);
    figure(figNum);
    [~,idx] = max(classScore);
    decisionIdx = (idx==iClass);
    imageClass = zeros(r,c);
    imageClass(fIdx) = decisionIdx(truthSameClass);
    imageClass(bIdx) = decisionIdx(truthDiffClass);
    imagesc(imageClass);
    title(sprintf('Class %d classification results (Yellow for classified)', iClass));
    saveas(figNum, [pnFigures filesep sprintf('classified_image_%d', iClass)], 'bmp');

    if iClass == 1
        figNum = 122+4*(iClass-1);
        figure(figNum);
        imageTruth = zeros(r,c);
        imageTruth(fIdx) = truthSameClass(truthSameClass);
        imagesc(imageTruth);
        title(sprintf('Truth Class %d', iClass));
        
        figNum = 123+4*(iClass-1);
        if ishandle(figNum), close(figNum); end
        figure(figNum);
        imageRes = zeros(r,c);
        % 1: true positive
        % 2: true negative
        % 3: false positive
        % 4: false negative
        imageRes(imageTruth == imageClass & imageClass == 1) = 1;
        imageRes(imageTruth == imageClass & imageClass == 0) = 2;
        imageRes(imageTruth ~= imageClass & imageClass == 1) = 3;        
        imageRes(imageTruth ~= imageClass & imageClass == 0) = 4;
        imagesc(imageRes); hold on;
        ax = axis;
        % 4 color map
        map = [0.1 0.8 0.1; ...
               0.5 0.5 0.5; ...
               0.8 0.1 0.1; ...
               0.0 0.3 0.9];
        colormap(map);
        % These fill nothing but generate handles for the legend with the correct colors
        h(1) = fill(1,1,map(1,:));
        h(2) = fill(1,1,map(2,:));
        h(3) = fill(1,1,map(3,:));
        h(4) = fill(1,1,map(4,:));
        legend(h, 'True Positive', 'True Negative', 'False Positive', 'False Negative', 'Location', 'eastoutside');
        axis(ax);
        title('Classification Results');
        p = get(gcf, 'Position');
        % increase width to fit legend
        set(gcf, 'Position', [p(1) p(2) p(3)*1.4 p(4)]);
        saveas(figNum, [pnFigures filesep sprintf('classifier_result_img')], 'bmp');
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
[histScores, histIndices] = hist(scoreDiff,linspace(minScore,maxScore,200));
pdfScores = histScores/sum(histScores(:));
cdfScores = cumsum(pdfScores);

minIndex = find(cdfScores < 0.01, 1, 'last');
if isempty(minIndex), minIndex = 1; end
maxIndex = find(cdfScores > 0.99, 1, 'first');
if isempty(maxIndex), maxIndex = 200; end
pdfDomain = linspace(histIndices(minIndex),histIndices(maxIndex),1000);

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

