%----------------------------------------------------------------
%
%  Program: generateSyntheticData
%
%  Purpose: produce samples of data from a Gaussian distribution
%
%  Programmer: Rod Pickens
%
%  Date: March 13, 2015
%
%----------------------------------------------------------------

function sampleData = generateSyntheticData(nSamples,meanV,covM)

%clc; close all; clear all; fclose('all');

fprintf(1,'Begin create samples for class\n');
tBeg = tic;

%-----------------------------------------------------------------
%
% Inputs
%
% 1) classifier identifier string
% classId = 2;
% 
% 2) train or test identifier value
% dataId    = 2; % 1 = train, 2 = test
% dataTypes = {'train','test'};
% 
% 3) output directory
% outputDir = 'project2\problem1';

% 4) Gaussian distribution parameters
% meanV = [4; 4];
% covM  = [1 0; 0 1];

%-----------------------------------------------------------------
% Generate the mapping matrix A (to color the data)
%
[V, D]=eig(covM);

A = V * sqrt(D);

% Input the number of samples to generate
% nSamples = 10000;

% % Enter the output file name
% dirName = sprintf('./data/classifier/project2/problem1/%s',dataTypes{dataId});
% fileNameData   = sprintf('%s/class%dsampleFeatures.dat',dirName,classId);
% fileNameParams = sprintf('%s/class%ddistParams.dat',dirName,classId);
% 
% fprintf(1,'data is %s\n',fileNameData);
% fprintf(1,'param file is %s\n',fileNameParams);

% Now create the data
sampleData = A*randn(numel(meanV),nSamples) + repmat(meanV,1,nSamples);

% % Write the data to the appropriate file
% csvwrite(fileNameData,sampleData);
% 
% % Write the parameters to appropriate file
% csvwrite(fileNameParams,[meanV(:)' covM(:)']);

fprintf(1,'Done create samples for class.  Time = %f (sec)\n',toc(tBeg));

end
