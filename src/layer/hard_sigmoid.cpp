#include "hard_sigmoid.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(HardSigmoid);

HardSigmoid::HardSigmoid() {}

HardSigmoid::~HardSigmoid() {}

Status HardSigmoid::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    // TODO: read from params
    alpha_ = 1.0f / 6.0f;
    beta_  = 0.5f;

    lower_ = -beta_ / alpha_;
    upper_ = lower_ + 1.0f / alpha_;

    return Status::kSuccess;
}

Status HardSigmoid::Validate() {
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
        LOG(ERROR) << "HardSigmoid::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    if (!IsSameShape(input_tensor_nodes_[0]->tensor.Shape(),
                     output_tensor_nodes_[0]->tensor.Shape())) {
        LOG(ERROR) << "HardSigmoid::Validate fail ["
                   << "error input/output shape"
                   << "]";
        return Status::kErrorShape;
    }

    return Status::kSuccess;
}

Status HardSigmoid::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const EigenTensorMap<float, 1> input_eigen_tensor =
        input.GetEigenTensor<float, 1>();

    EigenTensorMap<float, 1> output_eigen_tensor =
        output.GetEigenTensor<float, 1>();

    auto expr0 = input_eigen_tensor * alpha_ + beta_;

    auto expr1 = expr0.clip(0.0f, 1.0f);

    output_eigen_tensor.device(*device) = expr1;

    return Status::kSuccess;
}

}  // namespace SimpleInfer
