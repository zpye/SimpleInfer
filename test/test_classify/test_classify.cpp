#include <string>

#include "engine.h"
#include "logger.h"

using namespace SimpleInfer;

int main() {
    InitializeContext();

    const std::string model_path(MODEL_PATH);
    const std::string param_file =
        model_path + "/mobilenet/mobile_batch8.pnnx.param";
    const std::string bin_file = model_path + "/mobilenet/mobile_batch8.bin";
    int input_height           = 224;
    int input_width            = 224;

    Engine engine;
    engine.LoadModel(param_file, bin_file);

    // set input image data
    Tensor input(DataType::kFloat32, {8, input_height, input_width, 3}, true);
    EigenTensorMap<float, 4> input_eigen_tensor =
        input.GetEigenTensor<float, 4>();
    input_eigen_tensor.setConstant(2.0f);

    // inference
    engine.Input("0", input);
    engine.Forward();

    Tensor output;
    engine.Extract("122", output);

    EigenTensorMap<float, 2> result = output.GetEigenTensor<float, 2>();
    EigenDSize<2> result_shape      = result.dimensions();

    for (Eigen::DenseIndex b = 0; b < result_shape[0]; ++b) {
        float score_max             = -1.0f;
        Eigen::DenseIndex index_max = -1;

        for (Eigen::DenseIndex i = 0; i < result_shape[1]; ++i) {
            if (result(b, i) > score_max) {
                score_max = result(b, i);
                index_max = i;
            }
        }

        LOG(INFO) << absl::StrFormat("%d: (%d, %f)", b, index_max, score_max);
    }

    return 0;
}
