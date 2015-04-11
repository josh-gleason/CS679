    
%------------------------------------------------------------------
%
%  Program: 

    h=figure; hold on;
    plot(cdf_w1_hi,cdf_w0_hi,'b',cdf_w1_lo,cdf_w0_lo,'r'); grid on;
    title('ROC: p(d_M | w_0) versus p(d_M | w_1)');
    xlabel('p(d_M | w_1)');
    ylabel('p(d_M | w_0)');
    legend('high resolution','low resolution');
    savefig(h, [args.resultsdir filesep 'PartB_Performance.fig']);
    print(h, [args.resultsdir filesep 'PartB_Performance.png'], '-dpng')
    
    
    h=figure; hold on; 
    dv = 1:50;
    plot(dv,cmcH80(1:50),'b',...
         dv,cmcH90(1:50),'b-.',...
         dv,cmcH95(1:50),'b--',...
         dv,cmcL80(1:50),'r',...
         dv,cmcL90(1:50),'r-.',...
         dv,cmcL95(1:50),'r--');
    grid on;
    title('CMC for all experiments');
    xlabel('rank');
    ylabel('Performance');
    legend('High, Info = 0.80','High, Info = 0.90','High, Info = 0.95',...
           'Low, Info = 0.80','Low, Info = 0.90', 'Low, Info = 0.95');
