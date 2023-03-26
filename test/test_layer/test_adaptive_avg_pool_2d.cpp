#include "common.h"

#include "layer/adaptive_avg_pool_2d.h"

#include <algorithm>

TEST_CASE("Test AdaptiveAvgPool2d layer") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape{1, 8, 8, 3};
    std::vector<int> out_shape{1, 1, 1, 3};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    // global pooling

    // set layer
    AdaptiveAvgPool2d adaptive_avg_pool_2d_layer;
    adaptive_avg_pool_2d_layer.output_h_ = out_shape[1];
    adaptive_avg_pool_2d_layer.output_w_ = out_shape[2];

    CHECK_EQ(Status::kSuccess,
             adaptive_avg_pool_2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int l = 0; l < out_shape[3]; ++l) {
                    float value = 0.0f;
                    for (int h = 0; h < in_shape[1]; ++h) {
                        for (int w = 0; w < in_shape[2]; ++w) {
                            value += input_eigen_tensor(i, h, w, l);
                        }
                    }
                    value /= (in_shape[1] * in_shape[2]);

                    CHECK_FLOAT_EQ(output_eigen_tensor(i, j, k, l), value);
                }
            }
        }
    }
}
