function image = reproject_image(training, U, b)
%REPROJECT_IMAGE Reproject the image using coefficients in b
    image = reshape(U*b + training.mean, training.img_size);
end

