#include "binary_op.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(BinaryOp);

BinaryOp::BinaryOp() {}

BinaryOp::~BinaryOp() {}

Status BinaryOp::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "0", 2));
    switch (op->params.at("0").i) {
        case 0:
            binary_op_type_ = BinaryOpType::kAdd;
            break;
        case 2:
            binary_op_type_ = BinaryOpType::kMul;
            break;
        default:
            LOG(ERROR) << "unsupport BinaryOp type [" << op->params.at("0").i
                       << "]";
            return Status::kUnsupport;
    }

    return Status::kSuccess;
}

Status BinaryOp::Validate() {
    {
        Status ret = Layer::Validate();
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    {
        Status ret = ValidateShape(2, 1);
        if (Status::kSuccess != ret) {
            return ret;
        }
    }

    return Status::kSuccess;
}

Status BinaryOp::Forward(const std::vector<Tensor>& inputs, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const EigenTensorMap<float, 4> input0_eigen_tensor =
        inputs[0].GetEigenTensor<float, 4>();
    const EigenTensorMap<float, 4> input1_eigen_tensor =
        inputs[1].GetEigenTensor<float, 4>();
    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    const EigenDSize<4> input0_dsize = ToEigenDSize<4>(inputs[0].Shape());
    const EigenDSize<4> input1_dsize = ToEigenDSize<4>(inputs[1].Shape());
    const EigenDSize<4> output_dsize = ToEigenDSize<4>(output.Shape());

    const EigenDSize<4> input0_broadcast_dsize(
        output_dsize[0] / input0_dsize[0],
        output_dsize[1] / input0_dsize[1],
        output_dsize[2] / input0_dsize[2],
        output_dsize[3] / input0_dsize[3]);
    const EigenDSize<4> input1_broadcast_dsize(
        output_dsize[0] / input1_dsize[0],
        output_dsize[1] / input1_dsize[1],
        output_dsize[2] / input1_dsize[2],
        output_dsize[3] / input1_dsize[3]);

    auto expr0 = input0_eigen_tensor.broadcast(input0_broadcast_dsize);
    auto expr1 = input1_eigen_tensor.broadcast(input1_broadcast_dsize);

    switch (binary_op_type_) {
        case BinaryOpType::kAdd:
            output_eigen_tensor.device(*device) = expr0 + expr1;
            break;
        case BinaryOpType::kMul:
            output_eigen_tensor.device(*device) = expr0 * expr1;
            break;
        default:
            LOG(ERROR) << "unsupport BinaryOp type [" << (int)binary_op_type_
                       << "]";
            return Status::kUnsupport;
    }

    return Status::kSuccess;
}

Status BroadcastShape(const std::vector<int>& shape0,
                      const std::vector<int>& shape1,
                      std::vector<int>& broadcast_shape) {
    const int shape0_size = (int)shape0.size();
    const int shape1_size = (int)shape1.size();

    // TODO: support different shape size
    if (shape0_size != shape1_size) {
        LOG(ERROR) << "BroadcastShape: different shape size [" << shape0_size
                   << "][" << shape1_size << "]";
        return Status::kUnsupport;
    }

    broadcast_shape.resize(shape0_size);

    for (int i = 0; i < shape0_size; ++i) {
        if (shape0[i] == shape1[i]) {
            broadcast_shape[i] = shape0[i];
        } else if (1 == shape0[i]) {
            broadcast_shape[i] = shape1[i];
        } else if (1 == shape1[i]) {
            broadcast_shape[i] = shape0[i];
        } else {
            LOG(ERROR) << "BroadcastShape: different dim size [" << shape0[i]
                       << "][" << shape1[i] << "] at dimension [" << i << "]";
            return Status::kUnsupport;
        }
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
