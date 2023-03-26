#include "relu.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(ReLU);

ReLU::ReLU() {}

ReLU::~ReLU() {}

Status ReLU::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    return Status::kSuccess;
}

Status ReLU::Validate() {
    {
        Status ret = Layer::Validate();
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    {
        Status ret = ValidateShape(1, 1);
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    if (!(IsSameDataType<float>(input_tensor_nodes_[0]->tensor.GetDataType()) &&
          IsSameDataType<float>(
              output_tensor_nodes_[0]->tensor.GetDataType()))) {
        LOG(ERROR) << "ReLU::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    if (!IsSameShape(input_tensor_nodes_[0]->tensor.Shape(),
                     output_tensor_nodes_[0]->tensor.Shape())) {
        LOG(ERROR) << "ReLU::Validate fail ["
                   << "error input/output shape"
                   << "]";
        return Status::kErrorShape;
    }

    return Status::kSuccess;
}

Status ReLU::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const EigenTensorMap<float, 1> input_eigen_tensor =
        input.GetEigenTensor<float, 1>();

    EigenTensorMap<float, 1> output_eigen_tensor =
        output.GetEigenTensor<float, 1>();

    output_eigen_tensor.device(*device) = input_eigen_tensor.cwiseMax(0.0f);

    return Status::kSuccess;
}

}  // namespace SimpleInfer
