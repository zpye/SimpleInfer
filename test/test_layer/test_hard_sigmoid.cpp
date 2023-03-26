#include "common.h"

#include "layer/hard_sigmoid.h"

#include <algorithm>
#include <cmath>

static float HardSigmoidFunc(const float a,
                             const float alpha,
                             const float beta) {
    const float lower = -beta / alpha;
    const float upper = lower + 1.0f / alpha;

    if (a < lower) {
        return 0.0f;
    } else if (a > upper) {
        return 1.0f;
    }

    return (a * alpha + beta);
}

TEST_CASE("Test HardSigmoid layer") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> shape{1, 128, 128, 3};

    Tensor input_tensor(DataType::kFloat32, shape, true);
    Tensor output_tensor(DataType::kFloat32, shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    // set layer
    HardSigmoid hardsigmoid_layer;
    const float alpha = hardsigmoid_layer.alpha_;
    const float beta  = hardsigmoid_layer.beta_;
    CHECK_EQ(Status::kSuccess,
             hardsigmoid_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    CHECK_FLOAT_EQ(
                        output_eigen_tensor(i, j, k, l),
                        HardSigmoidFunc(input_eigen_tensor(i, j, k, l),
                                        alpha,
                                        beta));
                }
            }
        }
    }
}
