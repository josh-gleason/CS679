% Results

close all

% SVM Poly d=1 c=1
acc_lr(1) = 88.72;
% SVM Poly d=2 c=1
acc_lr(2) = 74.94;
% SVM Poly d=3 c=1
acc_lr(3) = 89.47;
% SVM Poly d=4 c=1
acc_lr(4) = 71.43;

% SVM RBF s=1 c=1
acc_lr(5) = 48.62;
% SVM RBF s=10 c=10
acc_lr(6) = 91.48;
% SVM RBF s=100 c=100
acc_lr(7) = 91.35;
% SVM RBF s=1000 c=1
acc_lr(8) = 88.97;

% Bayes
acc_lr(9) = 88.60;

% SVM Poly d=1 c=1
acc_hr(1) = 86.72;
% SVM Poly d=2 c=1
acc_hr(2) = 77.07;
% SVM Poly d=3 c=1
acc_hr(3) = 91.85;
% SVM Poly d=4 c=1
acc_hr(4) = 75.94;

% SVM RBF s=1 c=1
acc_hr(5) = 48.62;
% SVM RBF s=10 c=10
acc_hr(6) = 80.45;
% SVM RBF s=100 c=10
acc_hr(7) = 91.60;
% SVM RBF s=1000 c=1
acc_hr(8) = 48.62;

% Bayes
acc_hr(9) = 88.97;

lbl = {'Poly d=1', 'Poly d=2', 'Poly d=3', 'Poly d=4', 'RBF \sigma=1', 'RBF \sigma=10', 'RBF \sigma=100', 'RBF \sigma=1000', 'Bayesian'};

bar([acc_lr; acc_hr]');
ax = axis();
axis([ax(1) ax(2) 40 ax(4)]);
set(gca, 'XTickLabel', lbl);

rotateticklabel(gca, 70);

p = get(gca, 'Position');
p(2) = p(2) + 0.08;
p(4) = p(4) - 0.08;
set(gca, 'Position', p);
ylabel('Accuracy');
title('Classification Results');
legend('Low Res', 'High Res');