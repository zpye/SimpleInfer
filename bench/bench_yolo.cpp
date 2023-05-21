#include <benchmark/benchmark.h>

#include <string>

#include "engine.h"

static void BM_Yolov5s_Batch8_640x640(benchmark::State &state) {
    using namespace SimpleInfer;

    const std::string model_path(MODEL_PATH);
    const std::string param_file =
        model_path + "/yolo/demo/yolov5s_batch8.pnnx.param";
    const std::string bin_file =
        model_path + "/yolo/demo/yolov5s_batch8.pnnx.bin";

    InitializeContext();

    Engine engine;
    engine.LoadModel(param_file, bin_file);

    Tensor input(SimpleInfer::DataType::kFloat32, {8, 640, 640, 3}, true);
    engine.Input("0", input);
    engine.Forward();

    for (auto _ : state) {
        engine.Forward();
        // benchmark::DoNotOptimize(_);
    }

    Tensor output;
    engine.Extract("140", output);
}

BENCHMARK(BM_Yolov5s_Batch8_640x640)->Unit(benchmark::kMillisecond);
