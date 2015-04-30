SVM_CLASSIFIER   = 1;
BAYES_CLASSIFIER = 2;

NUM_FEATS = 30;

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

DEF_GAMMA = 1; %1/NUM_FEATS; %1/NUM_FEATS;
DEF_COEF0 = 0;
DEF_DEGREE = 1;
DEF_COST = 1;

% Try different parameters
resolution = HIGH_RES;
gamma = DEF_GAMMA;
coef0 = DEF_COEF0;
degree = DEF_DEGREE;
cost = DEF_COST;

degrange = [1 2 3 4];
sigrange = [1 10 100 1000];
gamrange = 1./(2*sigrange.^2);
costrange = [1 10 100 1000];

close all;
figidx = 0;

accuracy = zeros(length(costrange), 3);

% Test the Polynomial Kernel
for degree = degrange
    for cidx = 1:length(costrange)
        cost = costrange(cidx);
        accuracy(cidx,:) = main('classifier', SVM_CLASSIFIER, 'experiment', EXP_SVM_TRAIN_TEST, ...
            'resolution', resolution, 'kernel', SVM_KERNEL_POLY, ...
            'gamma', gamma, 'coef0', coef0, 'degree', degree, 'cost', cost, 'nfeats', NUM_FEATS);         
    end
    avg_acc = mean(accuracy,2);
    [accmax, idxmax] = max(avg_acc);
    costmax = costrange(idxmax);
    
    figidx = figidx + 1;
    figure(figidx); hold off;
    semilogx([costrange; costrange; costrange].', accuracy); hold on;
    semilogx(costrange, avg_acc, 'LineWidth', 2);
    axis([costrange(1) costrange(end) 45 100]);
    xlabel('Cost');
    ylabel('Accuracy');
    title(sprintf('SVM Results Polynomial Kernel of degree %d\nMax Average Accuracy %0.2f with cost %0.0f', degree, accmax, costmax));
    legend('Fold 1', 'Fold 2', 'Fold 3', 'Average', 'Location', 'Southeast');
end
degree = DEF_DEGREE;

% Test the RBF Kernel
for gamma = gamrange
    for cidx = 1:length(costrange)
        cost = costrange(cidx);
        accuracy(cidx,:) = main('classifier', SVM_CLASSIFIER, 'experiment', EXP_SVM_TRAIN_TEST, ...
            'resolution', resolution, 'kernel', SVM_KERNEL_RBF, ...
            'gamma', gamma, 'coef0', coef0, 'degree', degree, 'cost', cost, 'nfeats', NUM_FEATS);         
    end
    avg_acc = mean(accuracy,2);
    [accmax, idxmax] = max(avg_acc);
    costmax = costrange(idxmax);
    
    figidx = figidx + 1;
    figure(figidx); hold off;
    semilogx([costrange; costrange; costrange].', accuracy); hold on;
    semilogx(costrange, avg_acc, 'LineWidth', 2);
    axis([costrange(1) costrange(end) 45 100]);
    xlabel('Cost');
    ylabel('Accuracy');
    title(sprintf('SVM Results RBF Kernel with \\sigma %d\nMax Average Accuracy %0.2f with cost %0.0f', round(sqrt(1/(2*gamma))), accmax, costmax));
    legend('Fold 1', 'Fold 2', 'Fold 3', 'Average', 'Location', 'Southeast');
end
gamma = DEF_GAMMA;

% Bayesian classification
accuracy_bayes = main('classifier', BAYES_CLASSIFIER, 'experiment', EXP_BAYES_TRAIN_TEST, ...
            'resolution', resolution, 'nfeats', NUM_FEATS);

fprintf('Bayesian Classification Accuracy %0.2f%% %0.2f%% %0.2f%%\n', accuracy_bayes(1), accuracy_bayes(2), accuracy_bayes(3));
fprintf('Average Bayesian Classification Accuracy %0.2f%%\n', mean(accuracy_bayes));
