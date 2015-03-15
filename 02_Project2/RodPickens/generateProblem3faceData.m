%-------------------------------------------------------------
%
% Program: generateProblem3data
%
% Programmer: Rod Pickens
%
% Date: March 13, 2015
%
%-------------------------------------------------------------

function generateProblem3faceData()

%clear variables; close all; clc; fclose('all');

[fn, pn]=uigetfile('..\data\*','train image');
trainImg = imread([pn filesep fn]);

[fn, pn]=uigetfile('..\data\*','ref image');
refImg = imread([pn filesep fn]);

%---------------------------------------------------------------
%
% Extract the features for each class.  Classes are defined as:
%
%  Class id      class name
%    1             face
%    2             background (not face)
%
% The reference image is for training and contains the 
% truth, viz. the pixels in the image that are face and the 
% pixles that are not face.
%
[fFeats, bFeats] = extractFeatures(trainImg,refImg);

% 
nClasses = 2;
experiment = 'rgb';
tNow = datestr(now,30);
switch lower(experiment)
    case 'rgb'
        
        faceFeats = fFeats;
        bckgndFeats = bFeats;
        featFileName = sprintf('problem3_rgb_nC=%d_nF=%d_%s.mat',nClasses,3,tNow);
        %featFileName = sprintf('problem3_2classData_rgb_%s.mat',tNow);
        
    case 'chr'
        
        denomFace = fFeats(1,:)+fFeats(2,:)+fFeats(3,:)+0.1;
        faceFeats(1,:) = fFeats(1,:)./denomFace;
        faceFeats(2,:) = fFeats(2,:)./denomFace;
        
        denomBckgnd = bFeats(1,:)+bFeats(2,:)+bFeats(3,:)+0.1;
        bckgndFeats(1,:) = bFeats(1,:)./denomBckgnd;
        bckgndFeats(2,:) = bFeats(2,:)./denomBckgnd;
        
        featFileName = sprintf('problem3_chr_nC=%d_nF=%d_%s.mat',nClasses,2,tNow);
        %featFileName = sprintf('problem3_2classData_chrRG_%s.mat',tNow);
        
    case 'crcb'
        
        w = [-0.169 -0.332 0.5; 0.5 -0.419 -0.081];
        
        faceFeats   = w*fFeats;
        bckgndFeats = w*bFeats;
        
        featFileName = sprintf('problem3_CrBr_nC=%d_nF=%d_%s.mat',nClasses,2,tNow);
        %featFileName = sprintf('problem3_2classData_CrCb_%s.mat',tNow);
        
    otherwise
        error('no case correctly selected')
end

%
%
% structure for class statistics
nFeats = size(faceFeats,1); 
nClasses = 2;

% Create class definition structure
cp = struct('pC',NaN,'nS',NaN,'fV',NaN);
cDef = repmat(cp,nClasses,1);

% Class 1 description
% 
nFace = size(faceFeats,2); nBack = size(bckgndFeats,2);

cDef(1).pC = nFace/(nFace+nBack);
cDef(1).nS = size(faceFeats,2);
cDef(1).fV = faceFeats;

% Class 2 description
% 
cDef(2).pC = 1 - cDef(1).pC;
cDef(2).nS = size(bckgndFeats,2);
cDef(2).fV = bckgndFeats;

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
   
   features(:,iBeg:iEnd) = cDef(iClass).fV;
   truthData(1,iBeg:iEnd)    = repmat(iClass,1,nSamples);
   iBeg = iEnd + 1;
end

fprintf(1,'Save to feature directories related to %s\n',experiment);
pn = uigetdir('.','select directory to place project data (feature files)');

save([pn filesep featFileName],'-v7.3');

end

function [faceFeats, bckgndFeats] = extractFeatures(trainImg,refImg)
    
    binRef   = refImg(:,:,1);

    redImage = trainImg(:,:,1);
    grnImage = trainImg(:,:,2);
    bluImage = trainImg(:,:,3);

    indxFace   = find(binRef(:)==255);
    indxBckgnd = find(binRef(:)==0);

    faceFeats = double([redImage(indxFace)';...
                        grnImage(indxFace)';...
                        bluImage(indxFace)']);

    bckgndFeats = double([redImage(indxBckgnd)';...
                          grnImage(indxBckgnd)';...
                          bluImage(indxBckgnd)']);


end
