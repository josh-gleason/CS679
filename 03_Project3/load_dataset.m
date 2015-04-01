function [images, labels, img_size] = load_dataset(directory)
%LOAD_DATASET Loads all images into a column matrix and corresponding 
% labels. Also returns the original size of the images

    dstruct = dir(directory);
    fnames = dstruct(~[dstruct.isdir]);

    labels = zeros(1,length(fnames));
    for fidx = 1:length(fnames)
        fname = [directory filesep fnames(fidx).name];
        labels(fidx) = str2double(fnames(fidx).name(1:5));
        img = imread(fname);
        
        % store as column of the matrix A initialize if necessary
        if ~exist('images','var')
            images = zeros(numel(img), length(fnames));
            img_size = size(img);
        end
        
        % Also convert to floating point representation
        images(:,fidx) = double(img(:))/255;
    end

end

