#include "common.h"

#include "layer/conv_2d.h"

#include <algorithm>
#include <cmath>

TEST_CASE("Test Conv2d layer 0", "[Conv]") {
    using namespace SimpleInfer;

    const int in_image_height = 128;
    const int in_image_width  = 128;
    const int in_channel      = 32;
    const int out_channel     = 16;
    const int groups          = 1;
    const int kernel_h        = 3;
    const int kernel_w        = 3;
    const int stride_h        = 1;
    const int stride_w        = 1;
    const int dilation_h      = 1;
    const int dilation_w      = 1;
    const int padding_t       = 1;
    const int padding_b       = 1;
    const int padding_l       = 1;
    const int padding_r       = 1;

    const int k_h_size         = (kernel_h - 1) * dilation_h + 1;
    const int k_w_size         = (kernel_w - 1) * dilation_w + 1;
    const int out_image_height = (in_image_height + padding_t + padding_b -
                                  k_h_size + 1 + stride_h - 1) /
                                 stride_h;
    const int out_image_width =
        (in_image_width + padding_l + padding_r - k_w_size + 1 + stride_w - 1) /
        stride_w;
    const int start_h = -padding_t;
    const int start_w = -padding_l;

    // set tensor
    std::vector<int> in_shape{1, in_image_height, in_image_width, in_channel};
    std::vector<int> out_shape{1,
                               out_image_height,
                               out_image_width,
                               out_channel};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    // set layer
    Conv2d conv_2d_layer;
    conv_2d_layer.use_bias_     = true;
    conv_2d_layer.in_channels_  = in_channel;
    conv_2d_layer.out_channels_ = out_channel;
    conv_2d_layer.groups_       = groups;
    conv_2d_layer.kernel_h_     = kernel_h;
    conv_2d_layer.kernel_w_     = kernel_w;
    conv_2d_layer.stride_h_     = stride_h;
    conv_2d_layer.stride_w_     = stride_w;
    conv_2d_layer.dilation_h_   = dilation_h;
    conv_2d_layer.dilation_w_   = dilation_w;
    conv_2d_layer.padding_mode_ = Conv2d::PaddingMode::kZeros;
    conv_2d_layer.padding_t_    = padding_t;
    conv_2d_layer.padding_b_    = padding_b;
    conv_2d_layer.padding_l_    = padding_l;
    conv_2d_layer.padding_r_    = padding_r;

    EigenDSize<4> origin_shape(out_channel, in_channel, kernel_h, kernel_w);
    EigenDSize<4> shuffle_shape(kernel_h, kernel_w, in_channel, out_channel);

    EigenTensor<float, 4> origin_kernel(origin_shape);
    origin_kernel.setRandom();

    conv_2d_layer.weight_shape_ = shuffle_shape;
    conv_2d_layer.weight_.resize(shuffle_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 4> shuffle_kernel(
        reinterpret_cast<float*>(conv_2d_layer.weight_.data()),
        shuffle_shape);

    EigenDSize<4> shuffle(2, 3, 1, 0);
    shuffle_kernel = origin_kernel.shuffle(shuffle);

    EigenDSize<1> bias_shape(out_channel);
    conv_2d_layer.bias_shape_ = bias_shape;
    conv_2d_layer.bias_.resize(bias_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 1> bias_tensor(
        reinterpret_cast<float*>(conv_2d_layer.bias_.data()),
        bias_shape);
    bias_tensor.setRandom();

    CHECK_EQ(Status::kSuccess,
             conv_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int l = 0; l < out_shape[3]; ++l) {
                    float sum = 0.0f;
                    for (int c = 0; c < in_channel; ++c) {
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

                                sum +=
                                    input_eigen_tensor(i, input_h, input_w, c) *
                                    origin_kernel(l, c, h, w);
                            }
                        }
                    }

                    sum += bias_tensor(l);

                    const float out = output_eigen_tensor(i, j, k, l);
                    CHECK_FLOAT_EPS_EQ(out, sum, 2e-4);
                }
            }
        }
    }
}

TEST_CASE("Test Conv2d layer 1", "[Conv]") {
    using namespace SimpleInfer;

    const int in_image_height = 128;
    const int in_image_width  = 128;
    const int in_channel      = 32;
    const int out_channel     = 16;
    const int groups          = 2;
    const int kernel_h        = 3;
    const int kernel_w        = 3;
    const int stride_h        = 1;
    const int stride_w        = 1;
    const int dilation_h      = 1;
    const int dilation_w      = 1;
    const int padding_t       = 1;
    const int padding_b       = 1;
    const int padding_l       = 1;
    const int padding_r       = 1;

    const int k_h_size         = (kernel_h - 1) * dilation_h + 1;
    const int k_w_size         = (kernel_w - 1) * dilation_w + 1;
    const int out_image_height = (in_image_height + padding_t + padding_b -
                                  k_h_size + 1 + stride_h - 1) /
                                 stride_h;
    const int out_image_width =
        (in_image_width + padding_l + padding_r - k_w_size + 1 + stride_w - 1) /
        stride_w;
    const int start_h = -padding_t;
    const int start_w = -padding_l;

    // set tensor
    std::vector<int> in_shape{1, in_image_height, in_image_width, in_channel};
    std::vector<int> out_shape{1,
                               out_image_height,
                               out_image_width,
                               out_channel};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    const int in_channel_group  = in_channel / groups;
    const int out_channel_group = out_channel / groups;

    // set layer
    Conv2d conv_2d_layer;
    conv_2d_layer.use_bias_     = true;
    conv_2d_layer.in_channels_  = in_channel;
    conv_2d_layer.out_channels_ = out_channel;
    conv_2d_layer.groups_       = groups;
    conv_2d_layer.kernel_h_     = kernel_h;
    conv_2d_layer.kernel_w_     = kernel_w;
    conv_2d_layer.stride_h_     = stride_h;
    conv_2d_layer.stride_w_     = stride_w;
    conv_2d_layer.dilation_h_   = dilation_h;
    conv_2d_layer.dilation_w_   = dilation_w;
    conv_2d_layer.padding_mode_ = Conv2d::PaddingMode::kZeros;
    conv_2d_layer.padding_t_    = padding_t;
    conv_2d_layer.padding_b_    = padding_b;
    conv_2d_layer.padding_l_    = padding_l;
    conv_2d_layer.padding_r_    = padding_r;

    EigenDSize<4> origin_shape(out_channel,
                               in_channel_group,
                               kernel_h,
                               kernel_w);
    EigenDSize<4> shuffle_shape(kernel_h,
                                kernel_w,
                                in_channel_group,
                                out_channel);

    EigenTensor<float, 4> origin_kernel(origin_shape);
    origin_kernel.setRandom();

    conv_2d_layer.weight_shape_ = shuffle_shape;
    conv_2d_layer.weight_.resize(shuffle_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 4> shuffle_kernel(
        reinterpret_cast<float*>(conv_2d_layer.weight_.data()),
        shuffle_shape);

    EigenDSize<4> shuffle(2, 3, 1, 0);
    shuffle_kernel = origin_kernel.shuffle(shuffle);

    EigenDSize<1> bias_shape(out_channel);
    conv_2d_layer.bias_shape_ = bias_shape;
    conv_2d_layer.bias_.resize(bias_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 1> bias_tensor(
        reinterpret_cast<float*>(conv_2d_layer.bias_.data()),
        bias_shape);
    bias_tensor.setRandom();

    CHECK_EQ(Status::kSuccess,
             conv_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int g = 0; g < groups; ++g) {
                    for (int l = 0; l < out_channel_group; ++l) {
                        const int oc = g * out_channel_group + l;

                        float sum = 0.0f;
                        for (int c = 0; c < in_channel_group; ++c) {
                            const int ic = g * in_channel_group + c;
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

                                    sum += input_eigen_tensor(i,
                                                              input_h,
                                                              input_w,
                                                              ic) *
                                           origin_kernel(oc, c, h, w);
                                }
                            }
                        }

                        sum += bias_tensor(oc);

                        const float out = output_eigen_tensor(i, j, k, oc);
                        CHECK_FLOAT_EPS_EQ(out, sum, 2e-4);
                    }
                }
            }
        }
    }
}

TEST_CASE("Test Conv2d layer 2") {
    using namespace SimpleInfer;

    const int in_image_height = 640;
    const int in_image_width  = 640;
    const int in_channel      = 3;
    const int out_channel     = 32;
    const int groups          = 2;
    const int kernel_h        = 6;
    const int kernel_w        = 6;
    const int stride_h        = 2;
    const int stride_w        = 2;
    const int dilation_h      = 1;
    const int dilation_w      = 1;
    const int padding_t       = 2;
    const int padding_b       = 2;
    const int padding_l       = 2;
    const int padding_r       = 2;

    const int k_h_size         = (kernel_h - 1) * dilation_h + 1;
    const int k_w_size         = (kernel_w - 1) * dilation_w + 1;
    const int out_image_height = (in_image_height + padding_t + padding_b -
                                  k_h_size + 1 + stride_h - 1) /
                                 stride_h;
    const int out_image_width =
        (in_image_width + padding_l + padding_r - k_w_size + 1 + stride_w - 1) /
        stride_w;
    const int start_h = -padding_t;
    const int start_w = -padding_l;

    // set tensor
    std::vector<int> in_shape{8, in_image_height, in_image_width, in_channel};
    std::vector<int> out_shape{8,
                               out_image_height,
                               out_image_width,
                               out_channel};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    const int in_channel_group  = in_channel / groups;
    const int out_channel_group = out_channel / groups;

    // set layer
    Conv2d conv_2d_layer;
    conv_2d_layer.use_bias_     = true;
    conv_2d_layer.in_channels_  = in_channel;
    conv_2d_layer.out_channels_ = out_channel;
    conv_2d_layer.groups_       = groups;
    conv_2d_layer.kernel_h_     = kernel_h;
    conv_2d_layer.kernel_w_     = kernel_w;
    conv_2d_layer.stride_h_     = stride_h;
    conv_2d_layer.stride_w_     = stride_w;
    conv_2d_layer.dilation_h_   = dilation_h;
    conv_2d_layer.dilation_w_   = dilation_w;
    conv_2d_layer.padding_mode_ = Conv2d::PaddingMode::kZeros;
    conv_2d_layer.padding_t_    = padding_t;
    conv_2d_layer.padding_b_    = padding_b;
    conv_2d_layer.padding_l_    = padding_l;
    conv_2d_layer.padding_r_    = padding_r;

    EigenDSize<4> origin_shape(out_channel,
                               in_channel_group,
                               kernel_h,
                               kernel_w);
    EigenDSize<4> shuffle_shape(kernel_h,
                                kernel_w,
                                in_channel_group,
                                out_channel);

    EigenTensor<float, 4> origin_kernel(origin_shape);
    origin_kernel.setRandom();

    conv_2d_layer.weight_shape_ = shuffle_shape;
    conv_2d_layer.weight_.resize(shuffle_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 4> shuffle_kernel(
        reinterpret_cast<float*>(conv_2d_layer.weight_.data()),
        shuffle_shape);

    EigenDSize<4> shuffle(2, 3, 1, 0);
    shuffle_kernel = origin_kernel.shuffle(shuffle);

    EigenDSize<1> bias_shape(out_channel);
    conv_2d_layer.bias_shape_ = bias_shape;
    conv_2d_layer.bias_.resize(bias_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 1> bias_tensor(
        reinterpret_cast<float*>(conv_2d_layer.bias_.data()),
        bias_shape);
    bias_tensor.setRandom();

    CHECK_EQ(Status::kSuccess,
             conv_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int g = 0; g < groups; ++g) {
                    for (int l = 0; l < out_channel_group; ++l) {
                        const int oc = g * out_channel_group + l;

                        float sum = 0.0f;
                        for (int c = 0; c < in_channel_group; ++c) {
                            const int ic = g * in_channel_group + c;
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

                                    sum += input_eigen_tensor(i,
                                                              input_h,
                                                              input_w,
                                                              ic) *
                                           origin_kernel(oc, c, h, w);
                                }
                            }
                        }

                        sum += bias_tensor(oc);

                        const float out = output_eigen_tensor(i, j, k, oc);
                        CHECK_FLOAT_EPS_EQ(out, sum, 2e-4);
                    }
                }
            }
        }
    }
}

TEST_CASE("Test Conv2d layer 3", "[Conv]") {
    using namespace SimpleInfer;

    const int in_image_height = 10;
    const int in_image_width  = 10;
    const int in_channel      = 256;
    const int out_channel     = 255;
    const int groups          = 1;
    const int kernel_h        = 1;
    const int kernel_w        = 1;
    const int stride_h        = 1;
    const int stride_w        = 1;
    const int dilation_h      = 1;
    const int dilation_w      = 1;
    const int padding_t       = 0;
    const int padding_b       = 0;
    const int padding_l       = 0;
    const int padding_r       = 0;

    const int k_h_size         = (kernel_h - 1) * dilation_h + 1;
    const int k_w_size         = (kernel_w - 1) * dilation_w + 1;
    const int out_image_height = (in_image_height + padding_t + padding_b -
                                  k_h_size + 1 + stride_h - 1) /
                                 stride_h;
    const int out_image_width =
        (in_image_width + padding_l + padding_r - k_w_size + 1 + stride_w - 1) /
        stride_w;
    const int start_h = -padding_t;
    const int start_w = -padding_l;

    // set tensor
    std::vector<int> in_shape{4, in_image_height, in_image_width, in_channel};
    std::vector<int> out_shape{4,
                               out_image_height,
                               out_image_width,
                               out_channel};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    const int in_channel_group  = in_channel / groups;
    const int out_channel_group = out_channel / groups;

    // set layer
    Conv2d conv_2d_layer;
    conv_2d_layer.use_bias_     = true;
    conv_2d_layer.in_channels_  = in_channel;
    conv_2d_layer.out_channels_ = out_channel;
    conv_2d_layer.groups_       = groups;
    conv_2d_layer.kernel_h_     = kernel_h;
    conv_2d_layer.kernel_w_     = kernel_w;
    conv_2d_layer.stride_h_     = stride_h;
    conv_2d_layer.stride_w_     = stride_w;
    conv_2d_layer.dilation_h_   = dilation_h;
    conv_2d_layer.dilation_w_   = dilation_w;
    conv_2d_layer.padding_mode_ = Conv2d::PaddingMode::kZeros;
    conv_2d_layer.padding_t_    = padding_t;
    conv_2d_layer.padding_b_    = padding_b;
    conv_2d_layer.padding_l_    = padding_l;
    conv_2d_layer.padding_r_    = padding_r;

    EigenDSize<4> origin_shape(out_channel,
                               in_channel_group,
                               kernel_h,
                               kernel_w);
    EigenDSize<4> shuffle_shape(kernel_h,
                                kernel_w,
                                in_channel_group,
                                out_channel);

    EigenTensor<float, 4> origin_kernel(origin_shape);
    origin_kernel.setRandom();

    conv_2d_layer.weight_shape_ = shuffle_shape;
    conv_2d_layer.weight_.resize(shuffle_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 4> shuffle_kernel(
        reinterpret_cast<float*>(conv_2d_layer.weight_.data()),
        shuffle_shape);

    EigenDSize<4> shuffle(2, 3, 1, 0);
    shuffle_kernel = origin_kernel.shuffle(shuffle);

    EigenDSize<1> bias_shape(out_channel);
    conv_2d_layer.bias_shape_ = bias_shape;
    conv_2d_layer.bias_.resize(bias_shape.TotalSize() * sizeof(float));
    EigenTensorMap<float, 1> bias_tensor(
        reinterpret_cast<float*>(conv_2d_layer.bias_.data()),
        bias_shape);
    bias_tensor.setRandom();

    CHECK_EQ(Status::kSuccess,
             conv_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int g = 0; g < groups; ++g) {
                    for (int l = 0; l < out_channel_group; ++l) {
                        const int oc = g * out_channel_group + l;

                        float sum = 0.0f;
                        for (int c = 0; c < in_channel_group; ++c) {
                            const int ic = g * in_channel_group + c;
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

                                    sum += input_eigen_tensor(i,
                                                              input_h,
                                                              input_w,
                                                              ic) *
                                           origin_kernel(oc, c, h, w);
                                }
                            }
                        }

                        sum += bias_tensor(oc);

                        const float out = output_eigen_tensor(i, j, k, oc);
                        CHECK_FLOAT_EPS_EQ(out, sum, 2e-4);
                    }
                }
            }
        }
    }
}
