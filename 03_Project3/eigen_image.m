function b = eigen_image(training, U, image)
%EIGEN_IMAGE Project an image into a lower dimensional space
    image0 = image(:) - training.mean;
    b = (U.')*image0;
end

