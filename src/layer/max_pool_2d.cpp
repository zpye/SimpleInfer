#include "max_pool_2d.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(MaxPool2d);

MaxPool2d::MaxPool2d() {}

MaxPool2d::~MaxPool2d() {}

Status MaxPool2d::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "ceil_mode", 1));
    ceil_mode_ = op->params.at("ceil_mode").b;

    CHECK_BOOL(CheckParam(op, "return_indices", 1));
    return_indices_ = op->params.at("return_indices").b;

    CHECK_BOOL(CheckParam(op, "padding", 5));
    CHECK_BOOL(2 == op->params.at("padding").ai.size());
    if (2 == op->params.at("padding").ai.size()) {
        padding_t_ = padding_b_ = op->params.at("padding").ai[0];
        padding_l_ = padding_r_ = op->params.at("padding").ai[1];
    }

    CHECK_BOOL(CheckParam(op, "kernel_size", 5));
    CHECK_BOOL(2 == op->params.at("kernel_size").ai.size());
    kernel_h_ = op->params.at("kernel_size").ai[0];
    kernel_w_ = op->params.at("kernel_size").ai[1];

    CHECK_BOOL(CheckParam(op, "stride", 5));
    CHECK_BOOL(2 == op->params.at("stride").ai.size());
    stride_h_ = op->params.at("stride").ai[0];
    stride_w_ = op->params.at("stride").ai[1];

    CHECK_BOOL(CheckParam(op, "dilation", 5));
    CHECK_BOOL(2 == op->params.at("dilation").ai.size());
    dilation_h_ = op->params.at("dilation").ai[0];
    dilation_w_ = op->params.at("dilation").ai[1];

    return Status::kSuccess;
}

Status MaxPool2d::Validate() {
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
        LOG(ERROR) << "MaxPool2d::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status MaxPool2d::Forward(const Tensor& input, Tensor& output) {
    GET_EIGEN_THREADPOOL_DEVICE(device);

    const std::vector<int>& input_shape  = input.Shape();
    const std::vector<int>& output_shape = output.Shape();

    const int input_batch   = input_shape[0];
    const int input_height  = input_shape[1];
    const int input_width   = input_shape[2];
    const int input_channel = input_shape[3];

    const int output_batch   = output_shape[0];
    const int output_height  = output_shape[1];
    const int output_width   = output_shape[2];
    const int output_channel = output_shape[3];

    const EigenTensorMap<float, 4> input_eigen_tensor =
        input.GetEigenTensor<float, 4>();

    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    EigenDSize<2> reduce_dims(2, 3);
    EigenDSize<4> output_dims = ToEigenDSize<4>(output_shape);

    output_eigen_tensor.device(*device) =
        input_eigen_tensor
            .extract_image_patches(kernel_w_,
                                   kernel_h_,
                                   stride_w_,
                                   stride_h_,
                                   dilation_w_,
                                   dilation_h_,
                                   1,
                                   1,
                                   padding_t_,
                                   padding_b_,
                                   padding_l_,
                                   padding_r_,
                                   Eigen::NumTraits<float>::lowest())
            .maximum(reduce_dims)
            .reshape(output_dims);

    return Status::kSuccess;
}

}  // namespace SimpleInfer
