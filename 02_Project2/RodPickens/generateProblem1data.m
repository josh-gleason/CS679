%-------------------------------------------------------------
%
% Program: generateProblem1data
%
% Programmer: Rod Pickens
%
% Date: March 13, 2015
%
%-------------------------------------------------------------

clear all; close all; clc; fclose('all');

% structure for class statistics
nFeats = 2; nClasses = 3;
cp = struct('pC',NaN,'mV',NaN,'cM',NaN,'nS',NaN,'fV',NaN);
cDef = repmat(cp,nClasses,1);
nS = 100000;
% Class 1 description
% 
cDef(1).mV = [2; 2];
cDef(1).cM = [2  0; 0 2];
cDef(1).pC = 1/3;
cDef(1).nS = nS;
cDef(1).fV = zeros(nFeats,cDef(1).nS);

% Class 2 description
% 
cDef(2).mV = [1; 1];
cDef(2).cM = [0.5  0; 0  0.5];
cDef(2).pC = 1/3;
cDef(2).nS = nS;
cDef(2).fV = zeros(nFeats,cDef(2).nS);

% Class 3 description
% 
cDef(3).mV = [0; 0];
cDef(3).cM = [1  0; 0  1];
cDef(3).pC = 1/3;
cDef(3).nS = nS;
cDef(3).fV = zeros(nFeats,cDef(2).nS);

tNow = datestr(now,30);
featFileName = sprintf('problem1_nC=%d_nF=%d_nS=%d_%s.mat',nClasses,numel(cDef(2).mV),nS,tNow);

nTotalSamples = 0;
for iClass = 1:nClasses
    nTotalSamples = nTotalSamples + cDef(iClass).nS;
end

features = zeros(nFeats,nTotalSamples); 
truthData = zeros(1,nTotalSamples); 
iBeg = 1;
figure(5); cV = {'b.','g.','r.'};
for iClass = 1:nClasses
   % Generate the synthetic feature sets
   nSamples = cDef(iClass).nS;
   iEnd = iBeg + nSamples - 1;
   
   meanV    = cDef(iClass).mV;
   covM     = cDef(iClass).cM;
   
   cDef(iClass).fV = generateSyntheticData(nSamples,meanV,covM);
   features(:,iBeg:iEnd) = cDef(iClass).fV;
   plot(features(1,iBeg:iEnd),features(2,iBeg:iEnd),cV{iClass}); hold on; grid on;
   truthData(1,iBeg:iEnd)    = repmat(iClass,1,nSamples);
   iBeg = iEnd + 1;
end

pn = uigetdir('.','select directory to place project data');

save([pn filesep featFileName],'-v7.3');
