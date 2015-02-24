%----------------------------------------------------------
%
%  Program: chernoffError
%
%  Purpose: gives the argument to the Chernoff bound to probability of error
%
%     P(error) <= Chernoff Error Bound <= Bhatacharyya Error Bound
%
%  Inputs:
%
%     x = n-D feature vector
%     u = n-D mean vector for class i
%     S = n x n covariance matrix for class i
%
%  Outputs:
%
%    Chernoff Error Argument Bound
%
%  Programmer: Rod Pickens
%  
%  Date:  Feb 12, 2015
%
%-----------------------------------------------------------

function cError = chernoffErrorNormal(beta,classifierParams)
%    samples(iClass).classifierScore = bayesianClassifier(samples(iClass).features,classifierParams(iClass));
   
   meanV1 = classifierParams(1).meanV;
   covM1  = classifierParams(1).covM;
   meanV2 = classifierParams(2).meanV;
   covM2  = classifierParams(2).covM;
   
   cErrArg = beta*(1-beta)/2*(meanV1-meanV2)'*...
            ((1-beta)*covM1 + beta*covM2)^1*...
            (meanV1 - meanV2) + ...
            1/2*log( ((1-beta)*det(covM1) + beta*(det(covM2)))/( ...
            det(covM1)^(1-beta)*(det(covM2)^beta)));
  
   cError  = exp(-cErrArg);
   
   