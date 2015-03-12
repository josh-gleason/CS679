%---------------------------------------------------------------
%
%  Program: maxLikelihood
%
%  Purpose: Estimate the parameters of a probability distribution
%  function. For this project, we assume the samples used in the
%  maximum likelihood estimation come from a gaussian distribution.
%
%  Programmer: Rod Pickens
%
%  Date: March 11, 2015
%
%---------------------------------------------------------------

clc; close all; fclose('all'); clear all;

% Ask user to select the path and the directory of the data for which to
% perform maximum likelihood estimation.

pn = '.';
fn = 'prob_1a_featuresClass1.dat';

%---------------------------------------------------------------
% Class 1
meanc1f1 = 1; meanc1f2 = 1; 
stdc1f1f1 = 1; stdc1f2f2 = 1; 
rho = 0;
sc1f1f2 = rho * stdc1f1f1 * stdc1f2f2;
sc1f2f1 = sc1f1f2;
Cc1 = [stdc1f1f1^2    sc1f1f2;...
     sc1f2f1    stdc1f2f2^2];

[V, D]=eig(Cc1);

nSamples = 10000; nFeatures = 2;

meanV = repmat([meanc1f1; meanc1f2],1,nSamples);

seedR = rng;
xIn = randn(nFeatures,nSamples);
xOut = V*sqrt(D)*xIn + meanV;
figure; plot(xIn(1,:),xIn(2,:),'b.'); grid on;
figure; plot(xOut(1,:),xOut(2,:),'b.'); grid on;

csvwrite([pn filesep fn],xOut');

% Read the data from the selected file.
inDataC1 = csvread([pn filesep fn]);

%---------------------------------------------------------------
% Class 2
meanc2f1 = 4; meanc2f2 = 4; 
stdc2f1f1 = 1; stdc2f2f2 = 1; 
rho = 0.0;
sc2f1f2 = rho * stdc2f1f1 * stdc2f2f2;
sc2f2f1 = sc2f1f2;
Cc2 = [stdc2f1f1^2    sc2f1f2;...
     sc2f2f1    stdc2f2f2^2];

[V, D]=eig(Cc2);

nSamples = 10000; nFeatures = 2;

meanV = repmat([meanc2f1; meanc2f2],1,nSamples);

rng(seedR);
xIn = randn(nFeatures,nSamples);
xOut = V*sqrt(D)*xIn + meanV;
figure; plot(xIn(1,:),xIn(2,:),'b.'); grid on;
figure; plot(xOut(1,:),xOut(2,:),'b.'); grid on;

pn = '.';
fn = 'prob_1a_featuresClass2.dat';

csvwrite([pn filesep fn],xOut');

% Read the data from the selected file.
% file format structure.
% fprintf(fid,'p(w1), nfeatures, u11, u22, s1_11^2, s1_12; s1_21, s1_22^2)
% fprintf(fid,'p(w2), nfeatures, u21, u22, s2_11^2, s2_12; s2_21, s2_22^2)

p_w1 = 0.5;
p_w2 = 1 - p_w1;

iSkip = [1 2 4 8 10 100 1000];
nElements = numel(iSkip);

pn = '.';
fname = {'prob_1a_featuresClass1.dat','prob_1a_featuresClass2.dat'};

for iStep = 1:nElements
    % Determine how many samples to skip
    nSkip = iSkip(iStep);
    
    fn = fname{1};
    
    inData = csvread([pn filesep fn]);
    
    % get nsamples and ncols
    
    f1 = inData(1:nSkip:end,1);
    f2 = inData(1:nSkip:end,2);
    
    n4Estimate = numel(f1);
    
    fprintf(1,'iStep = %d and nSamples = %d\n',iStep,n4Estimate);
    
    fileOut = sprintf('classParams_nSamples_%d_nSkip_%d.dat',n4Estimate,nSkip);
    
    fid = fopen(fileOut,'w');
    
    c1u1 = mean(f1);
    c1u2 = mean(f2);
    
    c1c = cov(f1,f2);
    
    fn = fname{2};
    
    inData = csvread([pn filesep fn]);
    
    % get nsamples and ncols
    [nSamples, nFeatures]=size(inData);
    
    f1 = inData(1:nSkip:end,1);
    f2 = inData(1:nSkip:end,2);
    
    n4Estimate = numel(f1);
    c2u1 = mean(f1);
    c2u2 = mean(f2);
    
    c2c = cov(f1,f2);
    
    %fprintf(1,'nSamples = %d decimation factor=%d\n',n4Estimate,nSkip);
    
    fprintf(1,'\ttru class 1: u=<%f, %f> C=<%f, %f; %f, %f>\n',...
        meanc1f1,meanc1f2,Cc1(1,1),Cc1(1,2),Cc1(2,1),Cc1(2,2));    
    fprintf(1,'\test class 1: u=<%f, %f> C=<%f, %f; %f, %f>\n\n',...
        c1u1,c1u2,c1c(1,1),c1c(1,2),c1c(2,1),c1c(2,2));
    fprintf(1,'\ttru class 2: u=<%f, %f> C=<%f, %f; %f, %f>\n',...
        meanc2f1,meanc2f2,Cc2(1,1),Cc2(1,2),Cc2(2,1),Cc2(2,2));    
    fprintf(1,'\test class 2: u=<%f, %f> C=<%f, %f; %f, %f>\n',...
        c2u1,c2u2,c2c(1,1),c2c(1,2),c2c(2,1),c2c(2,2));
    
   
    fprintf(fid,'%f,%d,%f,%f,%f,%f,%f,%f\n',p_w1,nFeatures,c1u1,c1u2,c1c(1,1),c1c(1,2),c1c(2,1),c1c(2,2));
    
    fprintf(fid,'%f,%d,%f,%f,%f,%f,%f,%f\n',p_w2,nFeatures,c2u1,c2u2,c2c(1,1),c2c(1,2),c2c(2,1),c2c(2,2));
    
    fprintf(1,'\n');
end


    
