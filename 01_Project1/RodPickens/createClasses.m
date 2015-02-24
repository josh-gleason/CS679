%----------------------------------------------------------
%
%  Program: createClassSamples
%
%  Purpose: create n random samples given a model
%  for classification purposes.
%
%  Input:
%
%     a) nSamples
%     b) classParams: vector with mean and matrix with covariance
%
%  Output
%
%    a) matrix of features for each of n samples
%
%     features(1:Features,1:nSamples);
%
%  Programmer: Rod Pickens
%
%  Date: Feb 10, 2015
%
%----------------------------------------------------------
function samples = createClasses(nSamples,classifierParams)

[nFeatures, ~]=size(classifierParams.meanV);

mapA = mapping4distribution(classifierParams.covM);

meanV = repmat(classifierParams.meanV,1,nSamples);

% Test amd develop with the following matlab call and then
% use my own Gaussian numbers to check results.
%samples.features = mapA * randn(nFeatures,nSamples) + meanV;

% gaussianNumbers generates two sets of gaussian numbers 
% per call, so for this class, I will use as is.
%
if nFeatures ~= 2
    fprintf(1,'nFeatures = %d\n',nFeatures);
    error('need to modify script to accomodate gaussianNumber\n');
end
samples = mapA * gaussianNumbers(nSamples) + meanV;
end


function A = mapping4distribution(covM)

% Program:  mapping4distribution
%
% Purpose: Find the mathematical mapping from a normal uncorrelation 
%   gaussian distribution to a normal distribution with specified 
%   covariance.
%
%        v_white = D^(-1/2) * V' * v_nonNormal
%
% We are given v_white and want to compute v_nonNormal
%
%        v_nonNormal = V * D^(1/2) * v_white
%
% Where
%
%        [V, D] = eig(covM) and V is eigenVectors and D is eigenValues
%
% Find the eigenVectors and eigenValues of the covariance matrix
% 
% Input: covariance matrix
%
% Output: matrix A = V * sqrt(D);
%
% Programmer: Rod Pickens
%
[V, D]=eig(covM);

A = V * sqrt(D);

end




