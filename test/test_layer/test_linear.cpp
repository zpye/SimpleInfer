#include "common.h"

#include "layer/linear.h"

#include <algorithm>
#include <cmath>

TEST_CASE("Test Linear layer") {
    using namespace SimpleInfer;

    int in_features  = 128;
    int out_features = 64;

    // set tensor
    std::vector<int> in_shape{1, in_features};
    std::vector<int> out_shape{1, out_features};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 2> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 2>();
    EigenTensorMap<float, 2> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 2>();

    input_eigen_tensor.setRandom();

    // set layer
    Linear linear_layer;
    linear_layer.use_bias_     = true;
    linear_layer.in_features_  = in_features;
    linear_layer.out_features_ = out_features;

    linear_layer.weight_shape_ = EigenDSize<2>(out_features, in_features);
    linear_layer.weight_.resize(linear_layer.weight_shape_.TotalSize() *
                                sizeof(float));
    EigenTensorMap<float, 2> weight_tensor(
        reinterpret_cast<float*>(linear_layer.weight_.data()),
        linear_layer.weight_shape_);
    weight_tensor.setRandom();

    linear_layer.bias_shape_ = EigenDSize<1>(out_features);
    linear_layer.bias_.resize(linear_layer.bias_shape_.TotalSize() *
                              sizeof(float));
    EigenTensorMap<float, 1> bias_tensor(
        reinterpret_cast<float*>(linear_layer.bias_.data()),
        linear_layer.bias_shape_);
    bias_tensor.setRandom();

    CHECK_EQ(Status::kSuccess,
             linear_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            float value = 0.0f;
            for (int c = 0; c < in_shape[1]; ++c) {
                value += input_eigen_tensor(i, c) * weight_tensor(j, c);
            }
            value += bias_tensor(j);

            CHECK_FLOAT_EPS_EQ(output_eigen_tensor(i, j), value, 1e-4);
        }
    }
}
