%----------------------------------------------------------
%
%  Program: bayesianClassifier
%
%  Purpose: create a Bayesian probability score for the likelihood of a 
%  sample belonging to class wi given features x and model N(u,S).
%
%  Inputs:
%
%     x = n-D feature vector
%     classifierParams.u = n-D mean vector for class i
%     classifierParams.S = n x n covariance matrix for class i
%     classifierParams.pClass = pW = a-priori probability for class wi
%
%  Outputs:
%
%    bayesianScore
%
%  Math:
%    bayesianScore = ln (p(wi|x) + p(wi) )
%
%  Programmer: Rod Pickens
%  
%  Date:  Feb 10, 2015
%
%-----------------------------------------------------------

function bayesianScore = bayesianClassifier(x,classifierParams)
%    samples(iClass).classifierScore = bayesianClassifier(samples(iClass).features,classifierParams(iClass));

   [nFeatures, nSamples] = size(x);
   
   u  = repmat(classifierParams.meanV,1,nSamples);
   S  = classifierParams.covM;
   pW = classifierParams.pClass;
   
   invS = S\eye(size(S));
   
   weighting = (1/((2*pi)^(nFeatures/2))) * (1/sqrt(det(S)));
   
   gaussArg = 1/2*sum((x - u).*(invS * (x - u)));
   
   bayesianScore = log(weighting) + log(pW) - gaussArg;
   
   