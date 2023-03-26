#ifndef SIMPLE_INFER_INCLUDE_ENGINE_H_
#define SIMPLE_INFER_INCLUDE_ENGINE_H_

#include <string>

#include "tensor.h"
#include "types.h"

namespace SimpleInfer {

class EngineImpl;
class Engine {
public:
    Engine();

    ~Engine();

public:
    Status LoadModel(const std::string& parampath, const std::string& binpath);

    Status Release();

public:
    const std::vector<std::string> InputNames();
    const std::vector<std::string> OutputNames();

public:
    Status Input(const std::string& name, const Tensor& input);

    Status Forward();

    Status Extract(const std::string& name, Tensor& output);

private:
    EngineImpl* impl_ = nullptr;
};

void InitializeContext();

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_INCLUDE_ENGINE_H_
