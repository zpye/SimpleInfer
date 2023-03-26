#include <string>

#include "engine.h"

int main() {
    const std::string model_path(MODEL_PATH);
    const std::string param_file =
        model_path + "/batchnorm/resnet_batchnorm_sigmoid.pnnx.param";
    const std::string bin_file =
        model_path + "/batchnorm/resnet_batchnorm_sigmoid.pnnx.bin";
    // const std::string param_file =
    //     model_path + "/yolo/demo/yolov5s_batch8.pnnx.param";
    // const std::string bin_file =
    //     model_path + "/yolo/demo/yolov5s_batch8.pnnx.bin";

    SimpleInfer::InitializeContext();

    SimpleInfer::Engine engine;
    engine.LoadModel(param_file, bin_file);

    SimpleInfer::Tensor input(SimpleInfer::DataType::kFloat32,
                              {8, 640, 640, 3},
                              true);
    engine.Input("0", input);
    engine.Forward();

    SimpleInfer::Tensor output;
    engine.Extract("140", output);

    return 0;
}
