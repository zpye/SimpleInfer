#include "common.h"

#include "layer/upsample.h"

#include <algorithm>
#include <cmath>

TEST_CASE("Test Upsample layer 0", "[Upsample]") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape{1, 16, 16, 3};
    std::vector<int> out_shape{1, 32, 32, 3};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    const float scale_factor_h = (float)out_shape[1] / (float)in_shape[1];
    const float scale_factor_w = (float)out_shape[2] / (float)in_shape[2];

    // set layer
    Upsample upsample_layer;
    upsample_layer.upsample_mode_  = Upsample::UpsampleMode::kNearest;
    upsample_layer.scale_factor_h_ = scale_factor_h;
    upsample_layer.scale_factor_w_ = scale_factor_w;

    CHECK_EQ(Status::kSuccess,
             upsample_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            int h_in = (float)j / scale_factor_h;
            h_in     = (std::max)(0, (std::min)(in_shape[1] - 1, h_in));

            for (int k = 0; k < out_shape[2]; ++k) {
                int w_in = (float)k / scale_factor_w;
                w_in     = (std::max)(0, (std::min)(in_shape[2] - 1, w_in));

                for (int l = 0; l < out_shape[3]; ++l) {
                    CHECK_EQ(output_eigen_tensor(i, j, k, l),
                             input_eigen_tensor(i, h_in, w_in, l));
                }
            }
        }
    }
}

TEST_CASE("Test Upsample layer 1", "[Upsample]") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape{4, 10, 10, 128};
    std::vector<int> out_shape{4, 20, 20, 128};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    const float scale_factor_h = 2.0f;
    const float scale_factor_w = 2.0f;

    // set layer
    Upsample upsample_layer;
    upsample_layer.upsample_mode_  = Upsample::UpsampleMode::kNearest;
    upsample_layer.scale_factor_h_ = scale_factor_h;
    upsample_layer.scale_factor_w_ = scale_factor_w;

    CHECK_EQ(Status::kSuccess,
             upsample_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            int h_in = (float)j / scale_factor_h;
            h_in     = (std::max)(0, (std::min)(in_shape[1] - 1, h_in));

            for (int k = 0; k < out_shape[2]; ++k) {
                int w_in = (float)k / scale_factor_w;
                w_in     = (std::max)(0, (std::min)(in_shape[2] - 1, w_in));

                for (int l = 0; l < out_shape[3]; ++l) {
                    CHECK_EQ(output_eigen_tensor(i, j, k, l),
                             input_eigen_tensor(i, h_in, w_in, l));
                }
            }
        }
    }
}
