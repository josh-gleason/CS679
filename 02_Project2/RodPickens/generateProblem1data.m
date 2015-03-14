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

% Class 1 description
% 
cDef(1).mV = [0; 0];
cDef(1).cM = [1  0; 0 1];
cDef(1).pC = 0.5;
cDef(1).nS = 100;
cDef(1).fV = zeros(nFeats,cDef(1).nS);

% Class 2 description
% 
cDef(2).mV = [4; 4];
cDef(2).cM = [1  0; 0  1];
cDef(2).pC = 1 - cDef(1).pC;
cDef(2).nS = 100;
cDef(2).fV = zeros(nFeats,cDef(2).nS);

% Class 3 description
% 
cDef(3).mV = [-4; -4];
cDef(3).cM = [1  0; 0  1];
cDef(3).pC = 1 - cDef(1).pC;
cDef(3).nS = 100;
cDef(3).fV = zeros(nFeats,cDef(2).nS);

nTotalSamples = 0;
for iClass = 1:nClasses
    nTotalSamples = nTotalSamples + cDef(iClass).nS;
end

features = zeros(nFeats,nTotalSamples); 
truthData = zeros(1,nTotalSamples); 
iBeg = 1;
for iClass = 1:nClasses
   % Generate the synthetic feature sets
   nSamples = cDef(iClass).nS;
   iEnd = iBeg + nSamples - 1;
   
   meanV    = cDef(iClass).mV;
   covM     = cDef(iClass).cM;
   
   cDef(iClass).fV = generateSyntheticData(nSamples,meanV,covM);
   features(:,iBeg:iEnd) = cDef(iClass).fV;
   truthData(1,iBeg:iEnd)    = repmat(iClass,1,nSamples);
   iBeg = iEnd + 1;
end

pn = uigetdir('.','select directory to place project data');

save([pn filesep 'problem1_3classData_100samplesPerClass.mat'],'-v7.3');
