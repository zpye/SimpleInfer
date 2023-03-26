#include "common.h"

#include "layer/binary_op.h"

#include <algorithm>

TEST_CASE("Test BinaryOp layer [add]") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> shape{1, 128, 128, 3};

    Tensor input0_tensor(DataType::kFloat32, shape, true);
    Tensor input1_tensor(DataType::kFloat32, shape, true);
    Tensor output_tensor(DataType::kFloat32, shape, true);

    EigenTensorMap<float, 4> input0_eigen_tensor =
        input0_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> input1_eigen_tensor =
        input1_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input0_eigen_tensor.setRandom();
    input1_eigen_tensor.setRandom();

    // set layer
    BinaryOp binary_op_layer;
    binary_op_layer.binary_op_type_ = BinaryOp::BinaryOpType::kAdd;
    CHECK_EQ(
        Status::kSuccess,
        binary_op_layer.Forward({input0_tensor, input1_tensor}, output_tensor));

    // check
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    CHECK_FLOAT_EQ(output_eigen_tensor(i, j, k, l),
                                   input0_eigen_tensor(i, j, k, l) +
                                       input1_eigen_tensor(i, j, k, l));
                }
            }
        }
    }
}

TEST_CASE("Test BinaryOp layer [mul]") {
    using namespace SimpleInfer;

    // set tensor
    std::vector<int> shape{1, 128, 128, 3};

    Tensor input0_tensor(DataType::kFloat32, shape, true);
    Tensor input1_tensor(DataType::kFloat32, shape, true);
    Tensor output_tensor(DataType::kFloat32, shape, true);

    EigenTensorMap<float, 4> input0_eigen_tensor =
        input0_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> input1_eigen_tensor =
        input1_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input0_eigen_tensor.setRandom();
    input1_eigen_tensor.setRandom();

    // set layer
    BinaryOp binary_op_layer;
    binary_op_layer.binary_op_type_ = BinaryOp::BinaryOpType::kMul;
    CHECK_EQ(
        Status::kSuccess,
        binary_op_layer.Forward({input0_tensor, input1_tensor}, output_tensor));

    // check
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    CHECK_FLOAT_EQ(output_eigen_tensor(i, j, k, l),
                                   input0_eigen_tensor(i, j, k, l) *
                                       input1_eigen_tensor(i, j, k, l));
                }
            }
        }
    }
}
