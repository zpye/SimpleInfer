#include "common.h"

#include "layer/max_pool_2d.h"

#include <algorithm>

TEST_CASE("Test MaxPool2d layer 0") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape{1, 8, 8, 3};
    std::vector<int> out_shape{1, 4, 4, 3};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    // set layer
    int padding_t  = 0;
    int padding_b  = 0;
    int padding_l  = 0;
    int padding_r  = 0;
    int kernel_h   = in_shape[1] / out_shape[1];
    int kernel_w   = in_shape[2] / out_shape[2];
    int stride_h   = in_shape[1] / out_shape[1];
    int stride_w   = in_shape[2] / out_shape[2];
    int dilation_h = 1;
    int dilation_w = 1;

    MaxPool2d max_pool_2d_layer;
    max_pool_2d_layer.padding_t_  = padding_t;
    max_pool_2d_layer.padding_b_  = padding_b;
    max_pool_2d_layer.padding_l_  = padding_l;
    max_pool_2d_layer.padding_r_  = padding_r;
    max_pool_2d_layer.kernel_h_   = kernel_h;
    max_pool_2d_layer.kernel_w_   = kernel_w;
    max_pool_2d_layer.stride_h_   = stride_h;
    max_pool_2d_layer.stride_w_   = stride_w;
    max_pool_2d_layer.dilation_h_ = dilation_h;
    max_pool_2d_layer.dilation_w_ = dilation_w;

    CHECK_EQ(Status::kSuccess,
             max_pool_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int l = 0; l < out_shape[3]; ++l) {
                    float value = Eigen::NumTraits<float>::lowest();
                    for (int h = 0; h < kernel_h; ++h) {
                        for (int w = 0; w < kernel_w; ++w) {
                            value =
                                (std::max)(input_eigen_tensor(i,
                                                              j * stride_h + h,
                                                              k * stride_w + w,
                                                              l),
                                           value);
                        }
                    }

                    CHECK_EQ(output_eigen_tensor(i, j, k, l), value);
                }
            }
        }
    }
}

TEST_CASE("Test MaxPool2d layer 1") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape{8, 20, 20, 256};
    std::vector<int> out_shape{8, 20, 20, 256};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    // set layer
    int padding_t  = 2;
    int padding_b  = 2;
    int padding_l  = 2;
    int padding_r  = 2;
    int kernel_h   = 5;
    int kernel_w   = 5;
    int stride_h   = 1;
    int stride_w   = 1;
    int dilation_h = 1;
    int dilation_w = 1;

    const int start_h = -padding_t;
    const int start_w = -padding_l;

    MaxPool2d max_pool_2d_layer;
    max_pool_2d_layer.padding_t_  = padding_t;
    max_pool_2d_layer.padding_b_  = padding_b;
    max_pool_2d_layer.padding_l_  = padding_l;
    max_pool_2d_layer.padding_r_  = padding_r;
    max_pool_2d_layer.kernel_h_   = kernel_h;
    max_pool_2d_layer.kernel_w_   = kernel_w;
    max_pool_2d_layer.stride_h_   = stride_h;
    max_pool_2d_layer.stride_w_   = stride_w;
    max_pool_2d_layer.dilation_h_ = dilation_h;
    max_pool_2d_layer.dilation_w_ = dilation_w;

    CHECK_EQ(Status::kSuccess,
             max_pool_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int l = 0; l < out_shape[3]; ++l) {
                    float value = Eigen::NumTraits<float>::lowest();
                    for (int h = 0; h < kernel_h; ++h) {
                        for (int w = 0; w < kernel_w; ++w) {
                            int input_h =
                                start_h + j * stride_h + h * dilation_h;
                            int input_w =
                                start_w + k * stride_w + w * dilation_w;
                            if (input_h < 0 || input_h >= in_shape[1] ||
                                input_w < 0 || input_w >= in_shape[2]) {
                                continue;
                            }

                            value = (std::max)(
                                input_eigen_tensor(i, input_h, input_w, l),
                                value);
                        }
                    }

                    CHECK_EQ(output_eigen_tensor(i, j, k, l), value);
                }
            }
        }
    }
}
