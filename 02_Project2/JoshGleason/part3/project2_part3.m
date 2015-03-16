function project2_part3()
    close all

    % Query for data directory (default if canceled)
    dir = uigetdir('.', 'Select Directory with Training Data');
    if all(dir==0)
        dir = [cd filesep 'data'];
    end
    
    % Load training and test data (as floating point)
    images_rgb = cell(3,1);
    images_rgb{1} = double(imread([dir filesep 'Training_1.ppm']))/255.0;
    images_rgb{2} = double(imread([dir filesep 'Training_3.ppm']))/255.0;
    images_rgb{3} = double(imread([dir filesep 'Training_6.ppm']))/255.0;

    truth = cell(3,1);
    truth{1} = rgb_2_binary(imread([dir filesep 'ref1.ppm']), 1);
    truth{2} = rgb_2_binary(imread([dir filesep 'ref3.ppm']), 1);
    truth{3} = rgb_2_binary(imread([dir filesep 'ref6.ppm']), 1);

    % define which images are used for training and testing
    idx_train = 1;
    idx_test  = [2 3];

    [model_face_rgb, model_bg_rgb] = train_classifier(images_rgb{idx_train}, truth{idx_train});
    [roc_rgb, h_rgb] = test_classifier(model_face_rgb, model_bg_rgb, images_rgb(idx_test), truth(idx_test), 'RGB', '-r');
    
    % Convert from RGB to Chromatic color space (Ignoring luminance)
    images_chromatic = cell(3,1);
    images_chromatic{1} = rgb_2_chromatic(images_rgb{1});
    images_chromatic{2} = rgb_2_chromatic(images_rgb{2});
    images_chromatic{3} = rgb_2_chromatic(images_rgb{3});

    % Train and test classifier
    [model_face_chromatic, model_bg_chromatic] = train_classifier(images_chromatic{idx_train}, truth{idx_train});
    [roc_chromatic, h_chromatic] = test_classifier(model_face_chromatic, model_bg_chromatic, images_chromatic(idx_test), truth(idx_test), 'Chromatic', '-b');
    
    % Convert from RGB to CbCr color space (Ignoring luminance)
    images_cbcr = cell(3,1);
    images_cbcr{1} = rgb_2_cbcr(images_rgb{1});
    images_cbcr{2} = rgb_2_cbcr(images_rgb{2});
    images_cbcr{3} = rgb_2_cbcr(images_rgb{3});

    % Train and test classifier
    [model_face_cbcr, model_bg_cbcr] = train_classifier(images_cbcr{idx_train}, truth{idx_train});
    [roc_cbcr, h_cbcr] = test_classifier(model_face_cbcr, model_bg_cbcr, images_cbcr(idx_test), truth(idx_test), 'YCbCr', '-k');
    
    figure(10);
    legend([h_rgb(1) h_chromatic(1) h_cbcr(1) h_rgb(2) h_chromatic(2) h_cbcr(2)], 'RGB', 'Chromatic', 'YCbCr', ...
          'RGB Priors Threshold', 'Chromatic Priors Threshold', 'YCbCr Priors Threshold');
    title(sprintf('ROC Area\nRGB : %0.04f\nChromatic : %0.04f\nYCbCr : %0.04f', roc_rgb, roc_chromatic, roc_cbcr));
end

function [model_face, model_bg] = train_classifier(image, truth)
    % count number of dimensions in image
    [~,~,d] = size(image);

    % Extract training data from image (just calculate the x,y locations which are skin)
    face_idx = find(truth(:));
    bg_idx = find(~truth(:));
    n_face = length(face_idx);
    n_bg = length(bg_idx);

    % Build matrix of samples (d rows by n columns)
    x_face = zeros(d, n_face);
    x_bg = zeros(d, n_bg);
    for dim = 1:d
        % get a channel from the image then get labeled targets from that channel
        channel = image(:,:,dim);
        x_face(dim,:) = channel(face_idx);
        x_bg(dim,:) = channel(bg_idx);
    end

    model_face = struct('mean', [], 'covariance', []);
    model_bg = struct('mean', [], 'covariance', []);
    
    % Compute mean by summing across columns and dividing by n
    model_face.mean = sum(x_face,2) / n_face;
    model_bg.mean = sum(x_bg,2) / n_bg;
    % subtract the mean from X to compute covariance
    x0_face = bsxfun(@minus, x_face, model_face.mean);
    x0_bg = bsxfun(@minus, x_bg, model_bg.mean);
    % compute covariance
    model_face.covariance = x0_face*x0_face' / (n_face-1);
    model_bg.covariance = x0_bg*x0_bg' / (n_bg-1);
    
    % compute priors
    model_face.prior = n_face / (n_face + n_bg);
    model_bg.prior = 1 - model_face.prior;
end

function [roc_area, h] = test_classifier(model_face, model_bg, images, truth, title_str, line_type)
    % Test classifier (assumes all images are same size)
    [r,c,d] = size(images{1});
    n_images = length(images);

    % initialize matrix containing all test data and truth
    x = zeros(d, (r*c)*n_images);
    face_truth = false(1, (r*c)*n_images);

    % put all pixels from all images into x matrix
    for idx = 1:n_images
        image = images{idx};
        face = truth{idx};
        
        % region of x to add data to
        idx0 = (idx-1)*(r*c)+1;
        idx1 = (idx)*(r*c);
        
        % extract pixels and truth
        for dim = 1:d
            channel = image(:,:,dim);
            x(dim, idx0:idx1) = channel(:);
        end
        
        % store truth in single corresponding row vector
        face_truth(idx0:idx1) = face(:);
    end
    
    priors.p1 = model_face.prior;
    priors.p2 = model_bg.prior;
    params.sigma1 = model_face.covariance;
    params.sigma2 = model_bg.covariance;
    params.mu1 = model_face.mean;
    params.mu2 = model_bg.mean;

    % calculate probability density of each pixel using model
    g = discriminant(x, params, priors);

    % show an image of the probability of skin
    for idx = 1:n_images
        figure();
        
        % reshape into original image shape
        idx0 = (idx-1)*(r*c)+1;
        idx1 = (idx)*(r*c);
        imagesc(reshape(g(idx0:idx1), [r,c]));
        
        title(sprintf('Probability Density Image %s Test Image %d', title_str, idx));
        colorbar;
    end

     % compute ROC curves
     
    min_g = min(g(:));
    max_g = max(g(:));

    % Do I want to compute the ROC the easy way or hard way? Get same results both ways
    easy_roc = 1;
    if easy_roc
        range_g = linspace(min_g, max_g, 10000);
        pdfTruth = hist(g(face_truth), range_g);
        pdfBgnd = hist(g(~face_truth), range_g);
        pdfTruth = pdfTruth / sum(pdfTruth);
        pdfBgnd = pdfBgnd / sum(pdfBgnd);

        cdfTruth = cumsum(pdfTruth);
        cdfBgnd = cumsum(pdfBgnd);

        fpr = 1-cdfBgnd;
        fnr = cdfTruth;
    else
        N_LOW = 50;
        N_HIGH = 50;
        N_TOTAL = N_LOW + N_HIGH + 1;
        fp = zeros(1,N_TOTAL);
        fn = zeros(1,N_TOTAL);

        % space the thresholds using log spacing
        idx = 1;
        thresh_range_high = logspace(log10(0.01),log10(max_g), N_HIGH);
        thresh_range_low = -fliplr(logspace(log10(0.01),log10(-min_g), N_LOW));
        range_g = [thresh_range_low 0 thresh_range_high];
        for thresh = range_g;
            face = g > thresh;
            bg   = ~face;
            fp(idx) = sum(face(~face_truth)); % classified as skin but not actually skin
            fn(idx) = sum(bg(face_truth)); % not classified as skin but is actually skin
            idx = idx + 1;
        end

        fpr = fp / sum(~face_truth);
        fnr = fn / sum(face_truth);
    end

    % calculate a good threshold
    [~,thresh_idx] = min(fpr.^2+fnr.^2);
    fprintf('Good Threshold : %4.9f\n', range_g(thresh_idx));

    % find point on ROC closest to threshold = 0 used (based on priors)
    [~,prior_tidx] = min(abs(range_g));
    
    figure(10); hold on;
    h(1) = plot(fpr, fnr, line_type, 'LineWidth', 2);
    h(2) = plot(fpr(prior_tidx), fnr(prior_tidx), [' *' line_type(end)], 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('False Negative Rate');
    roc_area = trapz(fnr, fpr);
    axis([0 1 0 1]);
    grid on;
end

function y = mvgaussian_pdf(x, mu, cov)
    [n,~] = size(x);
    % subtract the mean
    x0 = bsxfun(@plus, -mu, x);
    % compute multivariate gaussian distribution
    y = 1/sqrt((2*pi)^n * det(cov)) * exp(-1/2 * sum(x0.*(cov\x0),1));
end

function image_bin = rgb_2_binary(image_rgb, thresh)
    [r,c,~] = size(image_rgb);
    
    % Combine into single channel
    image_grey = mean(image_rgb, 3);
    
    % create r by c boolean image
    image_bin = false(r,c);
    image_bin(image_grey > thresh) = true;
end

function image_cbcr = rgb_2_cbcr(image_rgb)
    [r,c,d] = size(image_rgb);
    
    assert(d==3);
    assert(isa(image_rgb, 'double'));

    image_cbcr = zeros(r,c,2);
    % cb = -0.169*R - 0.332*G + 0.500*B
    image_cbcr(:,:,1) = -0.169*image_rgb(:,:,1) - 0.332*image_rgb(:,:,2) + 0.500*image_rgb(:,:,3);
    % cr =  0.500*R - 0.419*G - 0.081*B
    image_cbcr(:,:,2) =  0.500*image_rgb(:,:,1) - 0.419*image_rgb(:,:,2) - 0.081*image_rgb(:,:,3);
end

function image_chromatic = rgb_2_chromatic(image_rgb)
    [r,c,d] = size(image_rgb);
    
    assert(d==3);
    assert(isa(image_rgb, 'double'));

    image_chromatic = zeros(r,c,2);
    % r = R / (R + G + B)
    image_chromatic(:,:,1) = image_rgb(:,:,1) ./ sum(image_rgb, 3);
    % g = G / (R + G + B)
    image_chromatic(:,:,2) = image_rgb(:,:,2) ./ sum(image_rgb, 3);
    
    image_chromatic(isnan(image_chromatic)) = 0;
end
