#include "common.h"

#include "layer/batch_norm_2d.h"

#include <algorithm>
#include <cmath>

static float BatchNorm2dFunc(const float val,
                             const float mean,
                             const float var,
                             const float eps,
                             const float weight,
                             const float bias) {
    return ((val - mean) * std::sqrt(1.0f / (var + eps)) * weight + bias);
}

TEST_CASE("Test BatchNorm2d layer") {
    using namespace SimpleInfer;

    const int channel = 32;

    // set tensor
    std::vector<int> shape{1, 128, 128, channel};

    Tensor input_tensor(DataType::kFloat32, shape, true);
    Tensor output_tensor(DataType::kFloat32, shape, true);

    EigenTensorMap<float, 4> input_eigen_tensor =
        input_tensor.GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output_tensor.GetEigenTensor<float, 4>();

    input_eigen_tensor.setRandom();

    // set layer
    BatchNorm2d batchnorm2d_layer;
    batchnorm2d_layer.eps_          = 1e-5;
    batchnorm2d_layer.use_affine_   = true;
    batchnorm2d_layer.num_features_ = channel;

    const int param_size = channel * sizeof(float);
    const EigenDSize<1> param_shape(channel);

    batchnorm2d_layer.running_mean_shape_ = param_shape;
    batchnorm2d_layer.running_mean_.resize(param_size);
    EigenTensorMap<float, 1> running_mean_tensor(
        reinterpret_cast<float*>(batchnorm2d_layer.running_mean_.data()),
        param_shape);
    running_mean_tensor.setRandom();

    batchnorm2d_layer.running_var_shape_ = param_shape;
    batchnorm2d_layer.running_var_.resize(param_size);
    EigenTensorMap<float, 1> running_var_tensor(
        reinterpret_cast<float*>(batchnorm2d_layer.running_var_.data()),
        param_shape);
    running_var_tensor = running_var_tensor.setRandom().square();

    batchnorm2d_layer.weight_shape_ = param_shape;
    batchnorm2d_layer.weight_.resize(param_size);
    EigenTensorMap<float, 1> weight_tensor(
        reinterpret_cast<float*>(batchnorm2d_layer.weight_.data()),
        param_shape);
    weight_tensor.setRandom();

    batchnorm2d_layer.bias_shape_ = param_shape;
    batchnorm2d_layer.bias_.resize(param_size);
    EigenTensorMap<float, 1> bias_tensor(
        reinterpret_cast<float*>(batchnorm2d_layer.bias_.data()),
        param_shape);
    bias_tensor.setRandom();

    CHECK_EQ(Status::kSuccess,
             batchnorm2d_layer.Forward(input_tensor, output_tensor));

    // check
    for (int i = 0; i < shape[0]; ++i) {
        for (int j = 0; j < shape[1]; ++j) {
            for (int k = 0; k < shape[2]; ++k) {
                for (int l = 0; l < shape[3]; ++l) {
                    CHECK_FLOAT_EPS_EQ(
                        output_eigen_tensor(i, j, k, l),
                        BatchNorm2dFunc(input_eigen_tensor(i, j, k, l),
                                        running_mean_tensor(l),
                                        running_var_tensor(l),
                                        batchnorm2d_layer.eps_,
                                        weight_tensor(l),
                                        bias_tensor(l)),
                        1e-4);
                }
            }
        }
    }
}
