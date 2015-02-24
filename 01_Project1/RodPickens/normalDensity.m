%----------------------------------------------------------
%
%  Program: normalDensity
%
%  Purpose: give the probability density value of a vector X
%
%  Inputs:
%
%     x = n-D feature vector
%     u = n-D mean vector for class i
%     S = n x n covariance matrix for class i
%
%  Outputs:
%
%    normalValue = p(x | normal density)
%
%  Programmer: Rod Pickens
%  
%  Date:  Feb 12, 2015
%
%-----------------------------------------------------------

function normalValue = normalDensity(x,classifierParams)
%    samples(iClass).classifierScore = bayesianClassifier(samples(iClass).features,classifierParams(iClass));

   [nFeatures, nSamples] = size(x);
   
   u  = repmat(classifierParams.meanV,1,nSamples);
   S  = classifierParams.covM;
   
   invS = S\eye(size(S));
   
   weighting = (1/((2*pi)^(nFeatures/2))) * (1/sqrt(det(S)));
   
   gaussArg = -1/2*sum((x - u).*(invS * (x - u)),1);
   
   normalValue = weighting * exp(gaussArg);
   