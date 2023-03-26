#include "linear.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(Linear);

Linear::Linear() {}

Linear::~Linear() {}

Status Linear::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "in_features", 2));
    in_features_ = op->params.at("in_features").i;

    CHECK_BOOL(CheckParam(op, "out_features", 2));
    out_features_ = op->params.at("out_features").i;

    CHECK_BOOL(CheckParam(op, "bias", 1));
    use_bias_ = op->params.at("bias").b;

    CHECK_BOOL(CheckAttr(op_, "weight", 1));
    weight_ = op_->attrs.at("weight").data;

    const std::vector<int>& weight_shape = op_->attrs.at("weight").shape;
    CHECK_BOOL(2 == weight_shape.size());
    // OI
    weight_shape_[0] = weight_shape[0];
    weight_shape_[1] = weight_shape[1];

    CHECK_BOOL(CheckAttr(op_, "bias", 1));
    bias_ = op_->attrs.at("bias").data;

    const std::vector<int>& bias_shape = op_->attrs.at("bias").shape;
    CHECK_BOOL(1 == bias_shape.size());
    bias_shape_[0] = bias_shape[0];

    return Status::kSuccess;
}

Status Linear::Validate() {
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
        LOG(ERROR) << "Linear::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status Linear::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const std::vector<int>& input_shape  = input.Shape();
    const std::vector<int>& output_shape = output.Shape();

    const int input_batch   = input_shape[0];
    const int input_channel = input_shape[1];

    const int output_batch   = output_shape[0];
    const int output_channel = output_shape[1];

    const EigenTensorMap<float, 2> input_eigen_tensor =
        input.GetEigenTensor<float, 2>();

    const EigenTensorMap<float, 2> weight_tensor(
        reinterpret_cast<float*>(weight_.data()),
        weight_shape_);

    EigenTensorMap<float, 2> output_eigen_tensor =
        output.GetEigenTensor<float, 2>();

    // shape
    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 1)};
    EigenDSize<2> output_dims(output_batch, output_channel);

    auto expr = input_eigen_tensor.contract(weight_tensor, contract_dims)
                    .reshape(output_dims);

    if (use_bias_) {
        const EigenTensorMap<float, 1> bias(
            reinterpret_cast<float*>(bias_.data()),
            bias_shape_);

        output_eigen_tensor.device(*device) =
            expr + bias.reshape(EigenDSize<2>(1, bias_shape_[0]))
                       .broadcast(EigenDSize<2>(output_batch, 1));
    } else {
        output_eigen_tensor.device(*device) = expr;
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
