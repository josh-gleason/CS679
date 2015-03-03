function project3()
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
    idx_test  = [2, 3];

    model_rgb = train_classifier(images_rgb{idx_train}, truth{idx_train});
    roc_rgb = test_classifier(model_rgb, images_rgb(idx_test), truth(idx_test), 'RGB', '-r');
    
    % Convert from RGB to Chromatic color space (Ignoring luminance)
    images_chromatic = cell(3,1);
    images_chromatic{1} = rgb_2_chromatic(images_rgb{1});
    images_chromatic{2} = rgb_2_chromatic(images_rgb{2});
    images_chromatic{3} = rgb_2_chromatic(images_rgb{3});

    % Train and test classifier
    model_chromatic = train_classifier(images_chromatic{idx_train}, truth{idx_train});
    roc_chromatic = test_classifier(model_chromatic, images_chromatic(idx_test), truth(idx_test), 'Chromatic', '-b');
    
    % Convert from RGB to CbCr color space (Ignoring luminance)
    images_cbcr = cell(3,1);
    images_cbcr{1} = rgb_2_cbcr(images_rgb{1});
    images_cbcr{2} = rgb_2_cbcr(images_rgb{2});
    images_cbcr{3} = rgb_2_cbcr(images_rgb{3});

    % Train and test classifier
    model_cbcr = train_classifier(images_cbcr{idx_train}, truth{idx_train});
    roc_cbcr = test_classifier(model_cbcr, images_cbcr(idx_test), truth(idx_test), 'YCbCr', '-k');
    
    figure(10);
    legend('RGB', 'Chromatic', 'YCbCr');
    title(sprintf('ROC Area\nRGB : %0.04f\nChromatic : %0.04f\nYCbCr : %0.04f', roc_rgb, roc_chromatic, roc_cbcr));
end

function model = train_classifier(image, truth)
    % count number of dimensions in image
    [~,~,d] = size(image);

    % Extract training data from image (just calculate the x,y locations which are skin)
    target_idx = find(truth(:));
    n = length(target_idx);

    % Build matrix of samples (d rows by n columns)
    x = zeros(d, n);
    for dim = 1:d
        % get a channel from the image then get labeled targets from that channel
        channel = image(:,:,dim);
        x(dim,:) = channel(target_idx);
    end

    model = struct('mean', [], 'covariance', []);
    
    % Compute mean by summing across columns and dividing by n
    model.mean = sum(x,2) / n;
    % subtract the mean from X to compute covariance
    x0 = bsxfun(@minus, x, model.mean);
    % compute covariance
    model.covariance = x0*x0' / (n-1);
end

function roc_area = test_classifier(model, images, truth, title_str, line_type)
    % Test classifier (assumes all images are same size)
    [r,c,d] = size(images{1});
    n_images = length(images);

    % initialize matrix containing all test data and truth
    x = zeros(d, (r*c)*n_images);
    t = false(1, (r*c)*n_images);

    % put all pixels from all images into x matrix
    for idx = 1:n_images
        image = images{idx};
        skin = truth{idx};
        
        % region of x to add data to
        idx0 = (idx-1)*(r*c)+1;
        idx1 = (idx)*(r*c);
        
        % extract pixels and truth
        for dim = 1:d
            channel = image(:,:,dim);
            x(dim, idx0:idx1) = channel(:);
        end
        
        % store truth in single corresponding row vector
        t(idx0:idx1) = skin(:);
    end
    
    % calculate probability density of each pixel using model
    p = mvgaussian_pdf(x, model.mean, model.covariance);
    
    % show an image of the probability of skin
    for idx = 1:n_images
        figure();
        
        % reshape into original image shape
        idx0 = (idx-1)*(r*c)+1;
        idx1 = (idx)*(r*c);
        imagesc(reshape(p(idx0:idx1), [r,c]));
        
        title(sprintf('Probability Density Image %s Test Image %d', title_str, idx));
        colorbar;
    end
    
    % compute ROC curves
    min_thresh = 0;
    max_thresh = 1/sqrt(det(model.covariance)*(2*pi)^d);
    idx = 1;
    fp = zeros(1,50);
    fn = zeros(1,50);
    % space the thresholds using log spacing
    thresh_range = fliplr(logspace(log10(min_thresh+1), log10(max_thresh+1), 50)-1);
    for thresh = thresh_range;
        skin = p>thresh;
        not_skin = ~skin;
        fp(idx) = sum(skin(~t)); % classified as skin but not actually skin
        fn(idx) = sum(not_skin(t)); % not classified as skin but is actually skin
        idx = idx + 1;
    end
    
    fpr = fp / sum(~t);
    fnr = fn / sum(t);
    
    figure(10); hold on;
    plot(fpr, fnr, line_type, 'LineWidth', 2);
    xlabel('False Positive Rate');
    ylabel('False Negative Rate');
    roc_area = trapz(fpr, fnr);
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
end
