#include "common.h"

#include "layer/flatten.h"

#include <algorithm>

TEST_CASE("Test Faltten layer") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape{1, 128, 2, 2};
    std::vector<int> out_shape{1, 128 * 2 * 2};

    Tensor input_tensor(DataType::kFloat32, in_shape, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 2> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 2>();

    input_eigen_tensor.setRandom();

    // set layer
    Flatten flatten_layer;
    flatten_layer.start_dim_ = 1;
    flatten_layer.end_dim_   = -1;

    CHECK_EQ(Status::kSuccess,
             flatten_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < in_shape[0]; ++i) {
        for (int j = 0; j < in_shape[1]; ++j) {
            for (int k = 0; k < in_shape[2]; ++k) {
                for (int l = 0; l < in_shape[3]; ++l) {
                    CHECK_EQ(output_eigen_tensor(i,
                                                 l * in_shape[1] * in_shape[2] +
                                                     j * in_shape[2] + k),
                             input_eigen_tensor(i, j, k, l));
                }
            }
        }
    }
}
