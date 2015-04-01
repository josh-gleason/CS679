function image = get_training_image(training, idx)
%GET_TRAINING_IMAGE Retrieve the training image from the training data
% struct
    image = reshape(training.samples(:,idx) + training.mean, training.img_size);
end

