%-------------------------------------------------------------
%   
%  Program: gaussianNumbers
%
%  Purpose:  generate Gaussian random number using the 
%    Box-Muller method and polar format 
%
%  http://www.design.caltech.edu/erik/Misc/Gaussian.html
%
%  Input
%     nSamples
%
%  Output: 
%
%     values

function rValues = gaussianNumbers(nSamples)

rValues = zeros(2,nSamples);
for iSample = 1:nSamples
    
   w = inf;
   while w >= 1.0
      x1 = 2.0 * rand - 1.0;
      x2 = 2.0 * rand - 1.0;
      w = x1 * x1 + x2 * x2;
    end;

    w = sqrt( (-2.0 * log( w ) ) / w );
    y1 = x1 * w;
    y2 = x2 * w;
    rValues(:,iSample) = [y1; y2];
    
end
  