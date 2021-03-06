// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>> training_set = {
    std::make_tuple(700, 161, 1, 32, 32, 20, 5, 0, 0, 2, 2),
    std::make_tuple(112, 112, 64, 16, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(224, 224, 3, 16, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1),
	std::make_tuple(240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1),
};

// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>> inference_server_set = {
    std::make_tuple(700, 161, 1, 1, 32, 20, 5, 0, 0, 2, 2),
    std::make_tuple(700, 161, 1, 2, 32, 20, 5, 0, 0, 2, 2),
    std::make_tuple(700, 161, 1, 4, 32, 20, 5, 0, 0, 2, 2),
    std::make_tuple(341, 79, 32, 1, 32, 10, 5, 0, 0, 2, 2),
    std::make_tuple(341, 79, 32, 2, 32, 10, 5, 0, 0, 2, 2),
    std::make_tuple(341, 79, 32, 4, 32, 10, 5, 0, 0, 2, 2),
    std::make_tuple(480, 48, 1, 1, 16, 3, 3, 1, 1, 1, 1),
    std::make_tuple(240, 24, 16, 1, 32, 3, 3, 1, 1, 1, 1),
    std::make_tuple(120, 12, 32, 1, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(60, 6, 64, 1, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(108, 108, 3, 1, 64, 3, 3, 1, 1, 2, 2),
    std::make_tuple(54, 54, 64, 1, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(27, 27, 128, 1, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(14, 14, 128, 1, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(7, 7, 256, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(224, 224, 3, 1, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(112, 112, 64, 1, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(56, 56, 128, 1, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(28, 28, 256, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(14, 14, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(224, 224, 3, 2, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(112, 112, 64, 2, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(56, 56, 128, 2, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(28, 28, 256, 2, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(14, 14, 512, 2, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(7, 7, 512, 2, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(224, 224, 3, 1, 64, 7, 7, 3, 3, 2, 2),
    std::make_tuple(28, 28, 192, 1, 32, 5, 5, 2, 2, 1, 1),
    std::make_tuple(28, 28, 192, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 512, 1, 48, 5, 5, 2, 2, 1, 1),
    std::make_tuple(14, 14, 512, 1, 192, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 832, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 832, 1, 128, 5, 5, 2, 2, 1, 1),
    std::make_tuple(224, 224, 3, 2, 64, 7, 7, 3, 3, 2, 2),
    std::make_tuple(28, 28, 192, 2, 32, 5, 5, 2, 2, 1, 1),
    std::make_tuple(28, 28, 192, 2, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 512, 2, 48, 5, 5, 2, 2, 1, 1),
    std::make_tuple(14, 14, 512, 2, 192, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 832, 2, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 832, 2, 128, 5, 5, 2, 2, 1, 1),
    std::make_tuple(56, 56, 64, 1, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(56, 56, 64, 1, 256, 1, 1, 0, 0, 2, 2),
    std::make_tuple(28, 28, 128, 1, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(28, 28, 128, 1, 512, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 256, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 256, 1, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 512, 1, 512, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 2048, 1, 512, 1, 1, 3, 3, 2, 2),
    std::make_tuple(56, 56, 64, 2, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(56, 56, 64, 2, 256, 1, 1, 0, 0, 2, 2),
    std::make_tuple(28, 28, 128, 2, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(28, 28, 128, 2, 512, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 256, 2, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 256, 2, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(14, 14, 256, 2, 1024, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 512, 2, 512, 1, 1, 0, 0, 1, 1),
    std::make_tuple(7, 7, 2048, 2, 512, 1, 1, 3, 3, 2, 2),
    std::make_tuple(700, 161, 1, 1, 64, 5, 5, 1, 1, 2, 2),
    std::make_tuple(350, 80, 64, 1, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(350, 80, 64, 1, 128, 5, 5, 1, 1, 2, 2),
    std::make_tuple(175, 40, 128, 1, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(175, 40, 128, 1, 256, 5, 5, 1, 1, 2, 2),
    std::make_tuple(84, 20, 256, 1, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(84, 20, 256, 1, 512, 5, 5, 1, 1, 2, 2),
    std::make_tuple(42, 10, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(700, 161, 1, 2, 64, 5, 5, 1, 1, 2, 2),
    std::make_tuple(350, 80, 64, 2, 64, 3, 3, 1, 1, 1, 1),
    std::make_tuple(350, 80, 64, 2, 128, 5, 5, 1, 1, 2, 2),
    std::make_tuple(175, 40, 128, 2, 128, 3, 3, 1, 1, 1, 1),
    std::make_tuple(175, 40, 128, 2, 256, 5, 5, 1, 1, 2, 2),
    std::make_tuple(84, 20, 256, 2, 256, 3, 3, 1, 1, 1, 1),
    std::make_tuple(84, 20, 256, 2, 512, 5, 5, 1, 1, 2, 2),
    std::make_tuple(42, 10, 512, 2, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(112, 112, 64, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 64, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 256, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 256, 1, 128, 1, 1, 0, 0, 2, 2),
    std::make_tuple(28, 28, 128, 1, 512, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 1, 128, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 1, 256, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 1, 1024, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 1024, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 1024, 1, 512, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(7, 7, 512, 1, 2048, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 1024, 1, 2048, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 2048, 1, 512, 1, 1, 0, 0, 1, 1),
    std::make_tuple(112, 112, 64, 2, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 64, 2, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 256, 2, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 256, 2, 128, 1, 1, 0, 0, 2, 2),
    std::make_tuple(28, 28, 128, 2, 512, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 2, 128, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 2, 256, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 256, 2, 1024, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 2, 1024, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 1024, 2, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 256, 2, 1024, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 1024, 2, 512, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 512, 2, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(7, 7, 512, 2, 2048, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 1024, 2, 2048, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 2048, 2, 512, 1, 1, 0, 0, 1, 1)
};

// Vector saves w, h, c, n, k, filter_w(s), filter_h(r), pad_w, pad_h, wstride, hstride
std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>> inference_device_set = {
    //std::make_tuple(151, 40, 1, 1, 32, 20, 5, 8, 8, 8, 2),  ARM convolution seg faults with this kernel
    std::make_tuple(112, 112, 64, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 64, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 256, 1, 64, 1, 1, 0, 0, 1, 1),
    std::make_tuple(56, 56, 256, 1, 128, 1, 1, 0, 0, 2, 2),
    std::make_tuple(28, 28, 128, 1, 512, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 1, 128, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 1, 256, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
    std::make_tuple(28, 28, 512, 1, 1024, 1, 1, 0, 0, 2, 2),
    std::make_tuple(14, 14, 1024, 1, 256, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 256, 1, 1024, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 1024, 1, 512, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 512, 1, 512, 3, 3, 1, 1, 1, 1),
    std::make_tuple(7, 7, 512, 1, 2048, 1, 1, 0, 0, 1, 1),
    std::make_tuple(14, 14, 1024, 1, 2048, 1, 1, 0, 0, 2, 2),
    std::make_tuple(7, 7, 2048, 1, 512, 1, 1, 0, 0, 1, 1)
};

