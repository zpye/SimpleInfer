#include "common.h"

#include "layer/cat.h"

#include <algorithm>

TEST_CASE("Test Cat layer [add]") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> in_shape0{1, 8, 8, 3};
    std::vector<int> in_shape1{1, 8, 8, 2};
    std::vector<int> in_shape2{1, 8, 8, 4};
    std::vector<int> out_shape{1, 8, 8, 9};

    Tensor input0_tensor(DataType::kFloat32, in_shape0, true);
    Tensor input1_tensor(DataType::kFloat32, in_shape1, true);
    Tensor input2_tensor(DataType::kFloat32, in_shape2, true);
    Tensor output_tensor(DataType::kFloat32, out_shape, true);

    EigenTensorMap<float, 4> input0_eigen_tensor =
        input0_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> input1_eigen_tensor =
        input1_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> input2_eigen_tensor =
        input2_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input0_eigen_tensor.setRandom();
    input1_eigen_tensor.setRandom();
    input2_eigen_tensor.setRandom();

    // set layer
    Cat cat_layer;
    cat_layer.dim_ = 1;  // channel

    CHECK_EQ(Status::kSuccess,
             cat_layer.Forward({input0_tensor, input1_tensor, input2_tensor},
                               output_tensor));

    // check
    for (int i = 0; i < out_shape[0]; ++i) {
        for (int j = 0; j < out_shape[1]; ++j) {
            for (int k = 0; k < out_shape[2]; ++k) {
                for (int l = 0; l < out_shape[3]; ++l) {
                    float value;
                    if (l < in_shape0[3]) {
                        value = input0_eigen_tensor(i, j, k, l);
                    } else if (l < in_shape0[3] + in_shape1[3]) {
                        value = input1_eigen_tensor(i, j, k, l - in_shape0[3]);
                    } else {
                        value = input2_eigen_tensor(
                            i,
                            j,
                            k,
                            l - (in_shape0[3] + in_shape1[3]));
                    }

                    CHECK_EQ(output_eigen_tensor(i, j, k, l), value);
                }
            }
        }
    }
}
