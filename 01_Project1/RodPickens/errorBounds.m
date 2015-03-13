%-------------------------------------------------------------
%
%  Program: errorBounds
%
%  Purpose: calculate Chernoff and Bhattacharya Error Bounds
%
%  Programmer: Rod Pickens
%
%  Date: Feb 10, 2015
%
%--------------------------------------------------------------

function errorBounds(classifierParams)

    bFactor = 0:0.05:1;
    
    pW1 = classifierParams(1).pClass;
    pW2 = classifierParams(2).pClass;
    
    pError = zeros(length(bFactor),1);
    
    chernoffErr = zeros(length(bFactor),1);
    for iB = 1:numel(bFactor)
        bV = bFactor(iB);
        pError(iB) = (pW1^bV)*(pW2^(1-bV))*df1*df2*sum(sum((pxW1.^(bV)).*(pxW2.^(1-bV))));
        chernoffErr(iB) = chernoffErrorNormal(bV, classifierParams);
    end
    
    bhattacharyaaErr = bhattacharyaaErrorNormal(classifierParams);
    
    fprintf(1,'P(error)=%f Chernoff bound=%f  Bhattcharyya bound=%f\n',...
        min(pError(:)),min(chernoffErr(:)),min(bhattacharyaaErr(:)));

    % Plot prob of error for varying beta.
    figure;
    plot(bFactor,pError,'b');
    title('Probability of Error');
    axis([0 1.1 0 1.1]); grid on;
    xlabel('\beta'); ylabel('P(error)');
