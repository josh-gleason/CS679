function test_data = load_test_data(testdir, mean_face)
%LOAD_TEST_DATA Load the test data and normalize samples by subtracting the
% mean face.

    [test_data.samples, test_data.labels, test_data.img_size] = load_dataset(testdir);
    test_data.mean = mean_face;
    test_data.samples = bsxfun(@minus, test_data.samples, test_data.mean);
end

