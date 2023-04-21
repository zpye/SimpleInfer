#include "layer.h"

namespace SimpleInfer {

Layer::Layer() {}

Layer::~Layer() {}

Status Layer::Init(const pnnx::Operator* op) {
    if (nullptr == op) {
        return Status::kEmpty;
    }

    op_ = op;

    return Status::kSuccess;
}

Status Layer::Init(const std::map<std::string, pnnx::Parameter>& params,
                   const std::map<std::string, pnnx::Attribute>& attrs) {
    return Status::kUnsupport;
}

void Layer::SetContext(Context* context) {
    context_ = context;
}

void Layer::SetInputNodes(const std::vector<TensorNode*>& input_tensor_nodes) {
    input_tensor_nodes_ = input_tensor_nodes;
}

void Layer::SetOutputNodes(
    const std::vector<TensorNode*>& output_tensor_nodes) {
    output_tensor_nodes_ = output_tensor_nodes;
}

Status Layer::Deinit() {
    return Status::kSuccess;
}

Status Layer::Validate() {
    return Status::kSuccess;
}

Status Layer::Forward() {
    LOG(INFO) << "Forward Layer [" << op_->name << "]";

    if (1 == input_tensor_nodes_.size()) {
        if (1 == output_tensor_nodes_.size()) {
            return Forward(input_tensor_nodes_[0]->tensor,
                           output_tensor_nodes_[0]->tensor);
        } else {
            std::vector<Tensor> outputs(output_tensor_nodes_.size());
            for (size_t i = 0; i < output_tensor_nodes_.size(); ++i) {
                outputs[i] = output_tensor_nodes_[i]->tensor;
            }

            return Forward(input_tensor_nodes_[0]->tensor, outputs);
        }
    } else {
        std::vector<Tensor> inputs(input_tensor_nodes_.size());
        for (size_t i = 0; i < input_tensor_nodes_.size(); ++i) {
            inputs[i] = input_tensor_nodes_[i]->tensor;
        }

        if (1 == output_tensor_nodes_.size()) {
            return Forward(inputs, output_tensor_nodes_[0]->tensor);
        } else {
            std::vector<Tensor> outputs(output_tensor_nodes_.size());
            for (size_t i = 0; i < output_tensor_nodes_.size(); ++i) {
                outputs[i] = output_tensor_nodes_[i]->tensor;
            }

            return Forward(inputs, outputs);
        }
    }

    return Status::kUnsupport;
}

Status Layer::Forward(const Tensor& input, Tensor& output) {
    return Status::kUnsupport;
}

Status Layer::Forward(const std::vector<Tensor>& inputs, Tensor& output) {
    return Status::kUnsupport;
}

Status Layer::Forward(const Tensor& input, std::vector<Tensor>& outputs) {
    return Status::kUnsupport;
}

Status Layer::Forward(const std::vector<Tensor>& inputs,
                      std::vector<Tensor>& outputs) {
    return Status::kUnsupport;
}

const pnnx::Operator* Layer::GetOp() {
    return op_;
}

Status Layer::ValidateShape(const int input_size, const int output_size) {
    if (input_size >= 0 && input_size != (int)input_tensor_nodes_.size()) {
        LOG(ERROR) << "ValidateShape fail ["
                   << "input size error " << (int)input_tensor_nodes_.size()
                   << ", need " << input_size << "]";
        return Status::kErrorShape;
    }

    if (output_size >= 0 && output_size != (int)output_tensor_nodes_.size()) {
        LOG(ERROR) << "ValidateShape fail ["
                   << "output size error " << (int)output_tensor_nodes_.size()
                   << ", need " << output_size << "]";
        return Status::kErrorShape;
    }

    return Status::kSuccess;
}

Eigen::ThreadPoolDevice* Layer::GetEigenThreadPoolDevice() {
    if (nullptr != context_) {
        return context_->GetEigenThreadPoolDevice();
    }

    return Context::GetDefaultEigenThreadPoolDevice();
}

}  // namespace SimpleInfer
