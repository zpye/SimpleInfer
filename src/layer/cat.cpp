#include "cat.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(Cat);

Cat::Cat() {}

Cat::~Cat() {}

Status Cat::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "dim", 2));
    dim_ = op->params.at("dim").i;

    return Status::kSuccess;
}

Status Cat::Validate() {
    {
        Status ret = Layer::Validate();
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    {
        Status ret = ValidateShape(-1, 1);
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    for (auto& input_tensor_node : input_tensor_nodes_) {
        if (!IsSameDataType<float>(input_tensor_node->tensor.GetDataType())) {
            LOG(ERROR) << "Cat::Validate fail ["
                       << "unsupport input data type"
                       << "]";
            return Status::kUnsupport;
        }
    }

    if (!IsSameDataType<float>(output_tensor_nodes_[0]->tensor.GetDataType())) {
        LOG(ERROR) << "Cat::Validate fail ["
                   << "unsupport output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status Cat::Forward(const std::vector<Tensor>& inputs, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const std::vector<int>& output_shape = output.Shape();

    // TODO: support
    if (4 != (int)output_shape.size()) {
        LOG(ERROR) << "Cat::Forward fail ["
                   << "unsupport output shape"
                   << "]";
        return Status::kUnsupport;
    }

    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    const int input_nums = (int)inputs.size();
    if (input_nums < 2) {
        LOG(ERROR) << "Cat::Forward fail ["
                   << "unsupport inputs size"
                   << "]";
        return Status::kUnsupport;
    }

    // NHWC
    int dim = dim_;
    if (1 == dim_) {
        dim = 3;
    } else if (2 == dim_) {
        dim = 1;
    } else if (3 == dim_) {
        dim = 2;
    }

    int offset = 0;
    for (int i = 0; i < input_nums; ++i) {
        auto input_i = inputs[i].GetEigenTensor<float, 4>();

        EigenDSize<4> input_dims = input_i.dimensions();
        EigenDSize<4> output_offset(0, 0, 0, 0);
        output_offset[dim] = offset;

        output_eigen_tensor.slice(output_offset, input_dims).device(*device) =
            input_i;

        offset += input_dims[dim];
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
