#include "engine.h"

#include "engine_impl.h"
#include "logger.h"

namespace SimpleInfer {

Engine::Engine() : impl_(new EngineImpl) {}

Engine::~Engine() {
    if (nullptr != impl_) {
        delete impl_;
        impl_ = nullptr;
    }
}

Status Engine::LoadModel(const std::string& parampath,
                         const std::string& binpath) {
    return impl_->LoadModel(parampath, binpath);
}

Status Engine::Release() {
    return impl_->Release();
}

const std::vector<std::string> Engine::InputNames() {
    return impl_->InputNames();
}

const std::vector<std::string> Engine::OutputNames() {
    return impl_->OutputNames();
}

Status Engine::Input(const std::string& name, const Tensor& input) {
    return impl_->Input(name, input);
}

Status Engine::Forward() {
    return impl_->Forward();
}

Status Engine::Extract(const std::string& name, Tensor& output) {
    return impl_->Extract(name, output);
}

void InitializeContext() {
    InitializeLogger();
}

}  // namespace SimpleInfer
