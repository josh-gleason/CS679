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
fn = 'featuresClass1.dat';

% Class 1
mean1   = 5; mean2   = 3; 
std11 =1.2; std22 = 0.8; 
rho = -0.8;
s12 = rho * std11 * std22;
s21 = s12;
C = [std11^2    s12;...
     s21    std22^2];

[V, D]=eig(C);

nSamples = 10000; nFeatures = 2;

meanV = repmat([mean1; mean2],1,nSamples);

xIn = randn(nFeatures,nSamples);
xOut = V*sqrt(D)*xIn + meanV;
figure; plot(xIn(1,:),xIn(2,:),'b.'); grid on;
figure; plot(xOut(1,:),xOut(2,:),'b.'); grid on;

csvwrite([pn filesep fn],xOut');
% Class 2
mean1   = 5; mean2   = 3; 
std11 =1.2; std22 = 0.8; 
rho = -0.8;
s12 = rho * std11 * std22;
s21 = s12;
C = [std11^2    s12;...
     s21    std22^2];

[V, D]=eig(C);

nSamples = 10000; nFeatures = 2;

meanV = repmat([mean1; mean2],1,nSamples);

xIn = randn(nFeatures,nSamples);
xOut = V*sqrt(D)*xIn + meanV;
figure; plot(xIn(1,:),xIn(2,:),'b.'); grid on;
figure; plot(xOut(1,:),xOut(2,:),'b.'); grid on;

pn = '.';
fn = 'featuresClass2.dat';

csvwrite([pn filesep fn],xOut');

% Read the data from the selected file.
inData = csvread([pn filesep fn]);

% get nsamples and ncols
[nSamples, nColumns]=size(inData);

% file format structure.
% fprintf(fid,'p(w1), nfeatures, u11, u22, s1_11^2, s1_12; s1_21, s1_22^2)
% fprintf(fid,'p(w2), nfeatures, u21, u22, s2_11^2, s2_12; s2_21, s2_22^2)

p_w1 = 0.5;
p_w2 = 1 - p_w1;

iSkip = [1 2 4 8 10 16 32 64];
nElements = numel(iSkip);

fprintf(1,'Truth: nf=%d, u=<%f, %f> C=<%f, %f; %f, %f>\n',...
        nColumns,mean1,mean2,C(1,1),C(1,2),C(2,1),C(2,2));

for iStep = 1:1 %nElements
    
    % Determine how many samples to skip
    nSkip = 2^iSkip(iStep);
    
    f1 = inData(1:nSkip:end,1);
    f2 = inData(1:nSkip:end,2);
    
    n4Estimate = numel(f1);
    u1 = mean(f1);
    u2 = mean(f2);
    
    c1 = cov(f1,f2);
    c2 = cov(f1,f2);
    
    fileOut = sprintf('cs679_proj2_experiment_%d_decimateFactor_%d.dat',iStep,nSkip)
    fid = fopen(fileOut,'w');
    
    fprintf(1,'p(w=%d)=%f, nf=%d, eu=<%f, %f> eC=<%f, %f; %f, %f>\n',...
        1,p_w1,nColumns,u1,u2,c1(1,1),c1(1,2),c1(2,1),c1(2,2));

    fprintf(fid,'%f,%d,%f,%f,%f,%f,%f,%f\n',p_w1,nColumns,u1,u2,c1(1,1),c1(1,2),c1(2,1),c1(2,2));
    
    fprintf(1,'p(w=%d)=%f, nf=%d, eu=<%f, %f> eC=<%f, %f; %f, %f>\n',...
        2,p_w1,nColumns,u1,u2,c1(1,1),c1(1,2),c1(2,1),c1(2,2));

    fprintf(fid,'%f,%d,%f,%f,%f,%f,%f,%f\n',p_w2,nColumns,u1,u2,c1(1,1),c1(1,2),c1(2,1),c1(2,2));

end

    
