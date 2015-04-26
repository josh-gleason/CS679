SVM_CLASSIFIER   = 1;
BAYES_CLASSIFIER = 2;

LOW_RES  = 1;
HIGH_RES = 2;

SVM_KERNEL_POLY = 1;
SVM_KERNEL_RBF  = 2;

EXP_SVM_TRAIN        = 1;
EXP_BAYES_TRAIN      = 2;
EXP_SVM_TEST         = 3;
EXP_BAYES_TEST       = 4;
EXP_SVM_TRAIN_TEST   = 5;
EXP_BAYES_TRAIN_TEST = 6;

NUM_FEATS = 134;

DEF_GAMMA = 1/2; %1/NUM_FEATS;
DEF_COEF0 = 1;
DEF_DEGREE = 3;
DEF_COST = 1;

% Try different parameters
resolution = LOW_RES;
gamma = DEF_GAMMA;
coef0 = DEF_COEF0;
degree = DEF_DEGREE;
cost = DEF_COST;

degrange = [1 2 3 4];
%gamrange = 1/NUM_FEATS * [0.1 0.3 0.6 1.0 1.4];
gamrange = 1./(2*[1 10 100 1000].^2);

costrange = 1:20:100;
costrange = 0.3:0.2:1.5
close all;
figidx = 0;

accuracy = zeros(length(costrange), 3);

% Test the Polynomial Kernel
for degree = degrange
    for cidx = 1:length(costrange)
        cost = costrange(cidx);
        accuracy(cidx,:) = main('classifier', SVM_CLASSIFIER, 'experiment', EXP_SVM_TRAIN_TEST, ...
            'resolution', resolution, 'kernel', SVM_KERNEL_POLY, ...
            'gamma', gamma, 'coef0', coef0, 'degree', degree, 'cost', cost);         
    end
    figidx = figidx + 1;
    figure(figidx); hold off;
    avg_acc = mean(accuracy,2);
    plot([costrange; costrange; costrange].', accuracy); hold on;
    plot(costrange, avg_acc, 'LineWidth', 2);
    axis([costrange(1) costrange(end) 45 100]);
    xlabel('Cost');
    ylabel('Accuracy');
    title(sprintf('SVM Results Polynomial Kernel of degree %d', degree));
    legend('Fold 1', 'Fold 2', 'Fold 3', 'Average', 'Location', 'Southeast');
end
degree = DEF_DEGREE;

% Test the RBF Kernel
for gamma = gamrange
    for cidx = 1:length(costrange)
        cost = costrange(cidx);
        accuracy(cidx,:) = main('classifier', SVM_CLASSIFIER, 'experiment', EXP_SVM_TRAIN_TEST, ...
            'resolution', resolution, 'kernel', SVM_KERNEL_RBF, ...
            'gamma', gamma, 'coef0', coef0, 'degree', degree, 'cost', cost);         
    end
    figidx = figidx + 1;
    figure(figidx); hold off;
    avg_acc = mean(accuracy,2);
    plot([costrange; costrange; costrange].', accuracy); hold on;
    plot(costrange, avg_acc, 'LineWidth', 2);
    axis([costrange(1) costrange(end) 45 100]);
    xlabel('Cost');
    ylabel('Accuracy');
    title(sprintf('SVM Results RBF Kernel with \\sigma %d', round(sqrt(1/(2*gamma)))));
    legend('Fold 1', 'Fold 2', 'Fold 3', 'Average', 'Location', 'Southeast');
end
gamma = DEF_GAMMA;
