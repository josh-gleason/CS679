function image = get_test_image(test_samples, training, idx)
%GET_TEST_IMAGE Retrieve the test image from the test_samples. Use training
% to construct correct size and unnormalize.
    image = reshape(test_samples(:,idx) + training.mean, training.img_size);
end

