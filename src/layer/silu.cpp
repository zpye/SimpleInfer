#include "silu.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(SiLU);

SiLU::SiLU() {}

SiLU::~SiLU() {}

Status SiLU::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    return Status::kSuccess;
}

Status SiLU::Validate() {
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
        LOG(ERROR) << "SiLU::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status SiLU::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const EigenTensorMap<float, 1> input_eigen_tensor =
        input.GetEigenTensor<float, 1>();

    EigenTensorMap<float, 1> output_eigen_tensor =
        output.GetEigenTensor<float, 1>();

    output_eigen_tensor.device(*device) =
        input_eigen_tensor / (1.0f + (-input_eigen_tensor).exp());

    return Status::kSuccess;
}

}  // namespace SimpleInfer
