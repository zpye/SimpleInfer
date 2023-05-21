#include "common.h"

#include "layer/conv_2d.h"

#include <algorithm>
#include <cmath>

void TestWinograd(const int in_image_height,
                  const int in_image_width,
                  const int in_channel,
                  const int out_channel,
                  const int groups,
                  const int kernel_h,
                  const int kernel_w,
                  const int stride_h,
                  const int stride_w,
                  const int dilation_h,
                  const int dilation_w,
                  const int padding_t,
                  const int padding_b,
                  const int padding_l,
                  const int padding_r) {
    using namespace SimpleInfer;

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
    // input_eigen_tensor.setConstant(1.0f);

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
    // origin_kernel.setConstant(1.0f);

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
    // bias_tensor.setConstant(1.0f);

    CHECK_EQ(Status::kSuccess, conv_2d_layer.InitWinograd());

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
                    CHECK_FLOAT_EPS_EQ(out, sum, 2e-3);

                    // LOG(INFO) << absl::StrFormat("(%d %d %d %d) (%f %f)",
                    //                              i,
                    //                              j,
                    //                              k,
                    //                              l,
                    //                              out,
                    //                              sum);
                }
            }
        }
    }
}

void TestWinograd(const int in_image_height,
                  const int in_image_width,
                  const int in_channel,
                  const int out_channel,
                  bool pad) {
    int padding = (pad ? 1 : 0);

    TestWinograd(in_image_height,
                 in_image_width,
                 in_channel,
                 out_channel,
                 1,
                 3,
                 3,
                 1,
                 1,
                 1,
                 1,
                 padding,
                 padding,
                 padding,
                 padding);
}

TEST_CASE("Test Winograd", "[Winograd]") {
    TestWinograd(4, 4, 1, 1, false);
    TestWinograd(4, 4, 1, 1, true);

    TestWinograd(4, 4, 2, 1, false);
    TestWinograd(4, 4, 4, 1, false);
    TestWinograd(4, 4, 7, 1, false);
    TestWinograd(4, 4, 2, 1, true);
    TestWinograd(4, 4, 4, 1, true);
    TestWinograd(4, 4, 7, 1, true);

    TestWinograd(4, 4, 1, 2, false);
    TestWinograd(4, 4, 1, 4, false);
    TestWinograd(4, 4, 1, 7, false);
    TestWinograd(4, 4, 1, 2, true);
    TestWinograd(4, 4, 1, 4, true);
    TestWinograd(4, 4, 1, 7, true);

    TestWinograd(4, 4, 2, 2, false);
    TestWinograd(4, 4, 4, 4, false);
    TestWinograd(4, 4, 7, 7, false);

    TestWinograd(4, 4, 1, 8, false);
    TestWinograd(4, 4, 3, 8, false);
    TestWinograd(4, 4, 8, 1, false);
    TestWinograd(4, 4, 8, 3, false);

    TestWinograd(4, 4, 128, 256, false);
    TestWinograd(4, 4, 256, 32, false);
    TestWinograd(4, 4, 32, 3, false);
}
