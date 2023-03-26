#include "adaptive_avg_pool_2d.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(AdaptiveAvgPool2d);

AdaptiveAvgPool2d::AdaptiveAvgPool2d() {}

AdaptiveAvgPool2d::~AdaptiveAvgPool2d() {}

Status AdaptiveAvgPool2d::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "output_size", 5));
    CHECK_BOOL(2 == op->params.at("output_size").ai.size());
    output_h_ = op->params.at("output_size").ai[0];
    output_w_ = op->params.at("output_size").ai[1];

    return Status::kSuccess;
}

Status AdaptiveAvgPool2d::Validate() {
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
        LOG(ERROR) << "AdaptiveAvgPool2d::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status AdaptiveAvgPool2d::Forward(const Tensor& input, Tensor& output) {
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

    // TODO: support
    if (0 != input_height % output_height || 0 != input_width % output_width) {
        LOG(ERROR) << "AdaptiveAvgPool2d::Forward fail ["
                   << "unsupport input/output shape"
                   << "]";
        return Status::kUnsupport;
    }

    const int kernel_h = input_height / output_height;
    const int kernel_w = input_width / output_width;

    const EigenTensorMap<float, 4> input_eigen_tensor =
        input.GetEigenTensor<float, 4>();

    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    EigenDSize<4> output_dims = ToEigenDSize<4>(output_shape);

    // global pooling
    if (1 == output_h_ && 1 == output_w_) {
        EigenDSize<2> reduce_dims(1, 2);
        output_eigen_tensor.device(*device) =
            input_eigen_tensor.mean(reduce_dims).reshape(output_dims);
    } else {
        EigenDSize<2> reduce_dims(2, 3);
        output_eigen_tensor.device(*device) =
            input_eigen_tensor
                .extract_image_patches(kernel_w,
                                       kernel_h,
                                       kernel_w,
                                       kernel_h,
                                       1,
                                       1,
                                       1,
                                       1,
                                       0,
                                       0,
                                       0,
                                       0,
                                       static_cast<float>(0))
                .mean(reduce_dims)
                .reshape(output_dims);
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
