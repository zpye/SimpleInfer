#include "flatten.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(Flatten);

Flatten::Flatten() {}

Flatten::~Flatten() {}

Status Flatten::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "start_dim", 2));
    start_dim_ = op->params.at("start_dim").i;

    CHECK_BOOL(CheckParam(op, "end_dim", 2));
    end_dim_ = op->params.at("end_dim").i;

    return Status::kSuccess;
}

Status Flatten::Validate() {
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
        LOG(ERROR) << "Flatten::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status Flatten::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const std::vector<int>& input_shape = input.Shape();
    const int input_dims_num            = (int)input_shape.size();

    int total_size = 1;
    for (int i = 0; i < input_dims_num; ++i) {
        total_size *= input_shape[i];
    }

    EigenDSize<1> output_dsize(total_size);
    EigenTensorMap<float, 1> output_eigen_tensor =
        output.GetEigenTensor<float, 1>();

    // TODO: support more
    if (4 == input_dims_num) {
        // NHWC -> NCHW
        EigenDSize<4> input_shuffle(0, 3, 1, 2);

        const EigenTensorMap<float, 4> input_eigen_tensor =
            input.GetEigenTensor<float, 4>();

        output_eigen_tensor.device(*device) =
            input_eigen_tensor.shuffle(input_shuffle).reshape(output_dsize);
    } else {
        const EigenTensorMap<float, 1> input_eigen_tensor =
            input.GetEigenTensor<float, 1>();

        output_eigen_tensor.device(*device) = input_eigen_tensor;
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
