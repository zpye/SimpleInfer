#include "batch_norm_2d.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(BatchNorm2d);

BatchNorm2d::BatchNorm2d() {}

BatchNorm2d::~BatchNorm2d() {}

Status BatchNorm2d::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "eps", 3));
    eps_ = op->params.at("eps").f;

    CHECK_BOOL(CheckParam(op, "num_features", 2));
    num_features_ = op->params.at("num_features").i;

    CHECK_BOOL(CheckParam(op, "affine", 1));
    use_affine_ = op->params.at("affine").b;

    CHECK_BOOL(CheckAttr(op_, "running_mean", 1));
    CHECK_BOOL(1 == op_->attrs.at("running_mean").shape.size());
    running_mean_shape_[0] = op_->attrs.at("running_mean").shape[0];
    running_mean_          = op_->attrs.at("running_mean").data;

    CHECK_BOOL(CheckAttr(op_, "running_var", 1));
    CHECK_BOOL(1 == op_->attrs.at("running_var").shape.size());
    running_var_shape_[0] = op_->attrs.at("running_var").shape[0];
    running_var_          = op_->attrs.at("running_var").data;

    CHECK_BOOL(CheckAttr(op_, "weight", 1));
    CHECK_BOOL(1 == op_->attrs.at("weight").shape.size());
    weight_shape_[0] = op_->attrs.at("weight").shape[0];
    weight_          = op_->attrs.at("weight").data;

    CHECK_BOOL(CheckAttr(op_, "bias", 1));
    CHECK_BOOL(1 == op_->attrs.at("bias").shape.size());
    bias_shape_[0] = op_->attrs.at("bias").shape[0];
    bias_          = op_->attrs.at("bias").data;

    return Status::kSuccess;
}

Status BatchNorm2d::Validate() {
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
        LOG(ERROR) << "BatchNorm2d::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    if (!IsSameShape(input_tensor_nodes_[0]->tensor.Shape(),
                     output_tensor_nodes_[0]->tensor.Shape())) {
        LOG(ERROR) << "BatchNorm2d::Validate fail ["
                   << "error input/output shape"
                   << "]";
        return Status::kErrorShape;
    }

    return Status::kSuccess;
}

Status BatchNorm2d::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const std::vector<int>& input_shape = input.Shape();

    const int input_batch   = input_shape[0];
    const int input_height  = input_shape[1];
    const int input_width   = input_shape[2];
    const int input_channel = input_shape[3];

    const EigenTensorMap<float, 4> input_eigen_tensor =
        input.GetEigenTensor<float, 4>();

    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    EigenTensorMap<float, 1> running_mean_tensor(
        reinterpret_cast<float*>(running_mean_.data()),
        running_mean_shape_);

    EigenTensorMap<float, 1> running_var_tensor(
        reinterpret_cast<float*>(running_var_.data()),
        running_var_shape_);

    EigenTensorMap<float, 1> weight_tensor(
        reinterpret_cast<float*>(weight_.data()),
        weight_shape_);

    EigenTensorMap<float, 1> bias_tensor(reinterpret_cast<float*>(bias_.data()),
                                         bias_shape_);

    // reshape
    EigenDSize<4> expand_shape(1, 1, 1, input_channel);
    EigenDSize<4> broadcast_factor(input_batch, input_height, input_width, 1);

    auto mean_expr =
        running_mean_tensor.reshape(expand_shape).broadcast(broadcast_factor);
    auto var_expr = (running_var_tensor + eps_)
                        .rsqrt()
                        .eval()
                        .reshape(expand_shape)
                        .broadcast(broadcast_factor);
    auto weight_expr =
        weight_tensor.reshape(expand_shape).broadcast(broadcast_factor);
    auto bias_expr =
        bias_tensor.reshape(expand_shape).broadcast(broadcast_factor);

    auto expr_out =
        (input_eigen_tensor - mean_expr) * var_expr * weight_expr + bias_expr;

    output_eigen_tensor.device(*device) = expr_out;

    return Status::kSuccess;
}

}  // namespace SimpleInfer
