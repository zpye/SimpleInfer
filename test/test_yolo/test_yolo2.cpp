#include <string>

#include "engine.h"
#include "logger.h"

using namespace SimpleInfer;

int main() {
    InitializeContext();

    const std::string model_path(MODEL_PATH);
    const std::string param_file =
        model_path + "/yolo/demo/yolov5n_small.pnnx.param";
    const std::string bin_file =
        model_path + "/yolo/demo/yolov5n_small.pnnx.bin";
    int input_height = 320;
    int input_width  = 320;

    Engine engine;
    engine.LoadModel(param_file, bin_file);

    // set input image data
    Tensor input(DataType::kFloat32, {4, input_height, input_width, 3}, true);
    EigenTensorMap<float, 4> input_eigen_tensor =
        input.GetEigenTensor<float, 4>();
    input_eigen_tensor.setConstant(42.0f);

    // inference
    engine.Input("0", input);
    engine.Forward();

    Tensor output;
    engine.Extract("140", output);

    EigenTensorMap<float, 3> result = output.GetEigenTensor<float, 3>();
    EigenDSize<3> result_shape      = result.dimensions();

    std::string output_line;
    for (int c = 0; c < result_shape[2]; ++c) {
        output_line =
            absl::StrAppendFormat(&output_line, "%.6f,", result(0, 0, c));
    }
    LOG(INFO) << output_line;

    return 0;
}
