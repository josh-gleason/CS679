function image = extract_image(data, idx)
%GET_TRAINING_IMAGE Get the image with the given index and add back in the mean.
% Assumes data contains a matrix of normalized samples (data.samples), the image
% size (data.img_size), and the mean image (data.mean)
    image = reshape(data.samples(:,idx) + data.mean, data.img_size);
end

