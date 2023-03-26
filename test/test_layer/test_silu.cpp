#include "common.h"

#include "layer/silu.h"

#include <algorithm>

TEST_CASE("Test SiLU layer") {
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
    SiLU silu_layer;
    CHECK_EQ(Status::kSuccess, silu_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    CHECK_FLOAT_EQ(
                        output_eigen_tensor(i, j, k, l),
                        input_eigen_tensor(i, j, k, l) /
                            (1.0f + std::exp(-input_eigen_tensor(i, j, k, l))));
                }
            }
        }
    }
}
