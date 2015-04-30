% Run some simple experiments with svm for demonstration

close all

KERNEL_LINEAR = 0;
KERNEL_POLY   = 1;
KERNEL_RBF    = 2;
KERNEL_SIG    = 3;

kernel = KERNEL_RBF;

sigma  = 0.5;
gamma  = 1/(2*sigma^2);
degree = 1;
cost   = 10000; 

if kernel == KERNEL_RBF
    kernel_str = 'RBF';
    param_str = ['\sigma = ' sprintf('%0.2f', sqrt(1/(2*gamma)))];
elseif kernel == KERNEL_POLY
    kernel_str = 'Polynomial';
    param_str = ['degree ' sprintf('%0.0f', degree)];
    gamma = 1;
end

data = [0 0 1; ...
        0 1 0.5];

theta = 10 * pi/180;
R = [cos(theta) sin(theta); ...
    -sin(theta) cos(theta)];

dataR = (R * data).';

labels = [1 1 2].';

params = sprintf('-s 0 -t %d -g %0.9f -d %d -c %0.4f', kernel, gamma, degree, cost);
model = svmtrain(labels, dataR, params);

mindims = min(dataR);
maxdims = max(dataR);
mindims = mindims - 0.5*(maxdims - mindims);
maxdims = maxdims + 0.5*(maxdims - mindims);

plot(dataR(labels == 1,1), dataR(labels == 1,2), 'r *', 'LineWidth', 2); hold on;
plot(dataR(labels == 2,1), dataR(labels == 2,2), 'b *', 'LineWidth', 2);
axis equal;

ax = axis();
xlims = ax(1:2);
ylims = ax(3:4);

xd = xlims(2) - xlims(1);
yd = ylims(2) - ylims(1);
xlims(1) = xlims(1) - 0.5*xd;
xlims(2) = xlims(2) + 0.5*xd;
ylims(1) = ylims(1) - 0.5*xd;
ylims(2) = ylims(2) + 0.5*xd;

ax = [xlims ylims];

[x,y] = meshgrid(linspace(xlims(1),xlims(2),500), linspace(ylims(1),ylims(2),500));
v = [x(:) y(:)];

colormap([1.0 0.7 0.7; 0.7 0.7 1.0]);
predictions = svmpredict(zeros(length(v),1), v, model);
predictions = reshape(predictions, 500, 500);

contourf(x, y, predictions, 1.5);

hold on;
plot(dataR(labels == 1,1), dataR(labels == 1,2), 'r *', 'LineWidth', 2);
plot(dataR(labels == 2,1), dataR(labels == 2,2), 'b *', 'LineWidth', 2);
title(['SVM Kernel ' kernel_str ' with ' param_str]);
axis(ax);