#include "conv_2d.h"

#include "simd/binary.h"
#include "simd/gemm.h"
#include "simd/parallel.h"
#include "simd/winograd_helper.h"

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(Conv2d);

Conv2d::Conv2d() {}

Conv2d::~Conv2d() {}

Status Conv2d::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    return Init(op->params, op->attrs);
}

Status Conv2d::Init(const std::map<std::string, pnnx::Parameter>& params,
                    const std::map<std::string, pnnx::Attribute>& attrs) {
    CHECK_BOOL(CheckParam(params, "padding_mode", 4));
    if ("zeros" == params.at("padding_mode").s) {
        padding_mode_ = PaddingMode::kZeros;
    } else if ("replicate" == params.at("padding_mode").s) {
        padding_mode_ = PaddingMode::kReplicate;
    } else if ("reflect" == params.at("padding_mode").s) {
        padding_mode_ = PaddingMode::kReflect;
    } else {
        LOG(ERROR) << "Conv2d::Init fail ["
                   << "unsupport padding mode"
                   << "]";
        return Status::kUnsupport;
    }

    CHECK_BOOL(CheckParam(params, "padding", 5));
    CHECK_BOOL(2 == params.at("padding").ai.size());
    if (2 == params.at("padding").ai.size()) {
        padding_t_ = padding_b_ = params.at("padding").ai[0];
        padding_l_ = padding_r_ = params.at("padding").ai[1];
    }

    CHECK_BOOL(CheckParam(params, "kernel_size", 5));
    CHECK_BOOL(2 == params.at("kernel_size").ai.size());
    kernel_h_ = params.at("kernel_size").ai[0];
    kernel_w_ = params.at("kernel_size").ai[1];

    CHECK_BOOL(CheckParam(params, "stride", 5));
    CHECK_BOOL(2 == params.at("stride").ai.size());
    stride_h_ = params.at("stride").ai[0];
    stride_w_ = params.at("stride").ai[1];

    CHECK_BOOL(CheckParam(params, "dilation", 5));
    CHECK_BOOL(2 == params.at("dilation").ai.size());
    dilation_h_ = params.at("dilation").ai[0];
    dilation_w_ = params.at("dilation").ai[1];

    CHECK_BOOL(CheckParam(params, "groups", 2));
    groups_ = params.at("groups").i;

    CHECK_BOOL(CheckParam(params, "in_channels", 2));
    in_channels_ = params.at("in_channels").i;

    CHECK_BOOL(CheckParam(params, "out_channels", 2));
    out_channels_ = params.at("out_channels").i;

    CHECK_STATUS(InitWeightAndBias(params, attrs));

    CHECK_STATUS(InitWinograd());

    return Status::kSuccess;
}

Status Conv2d::Validate() {
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
        LOG(ERROR) << "Conv2d::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

Status Conv2d::Forward(const Tensor& input, Tensor& output) {
    if (use_winograd_) {
        return ForwardWinograd23(input, output);
    }

    if (1 == groups_) {
        return ForwardIm2Col(input, output);
    }

    return ForwardIm2ColWithGroup(input, output);
}

Status Conv2d::InitWeightAndBias(
    const std::map<std::string, pnnx::Parameter>& params,
    const std::map<std::string, pnnx::Attribute>& attrs) {
    CHECK_BOOL(CheckAttr(attrs, "weight", 1));

    std::vector<int> weight_shape = attrs.at("weight").shape;
    std::vector<char> weight      = attrs.at("weight").data;

    CHECK_BOOL(4 == weight_shape.size());

    weight_.resize(weight.size());

    weight_shape_[0] = weight_shape[2];
    weight_shape_[1] = weight_shape[3];
    weight_shape_[2] = weight_shape[1];
    weight_shape_[3] = weight_shape[0];

    // weight
    EigenTensorMap<float, 4> weight_original(
        reinterpret_cast<float*>(weight.data()),
        EigenDSize<4>(weight_shape[0],
                      weight_shape[1],
                      weight_shape[2],
                      weight_shape[3]));
    EigenTensorMap<float, 4> weight_transform(
        reinterpret_cast<float*>(weight_.data()),
        weight_shape_);

    // OIHW -> HWIO
    EigenDSize<4> weight_shuffle(2, 3, 1, 0);
    weight_transform = weight_original.shuffle(weight_shuffle);

    CHECK_BOOL(CheckParam(params, "bias", 1));
    use_bias_ = params.at("bias").b;

    if (use_bias_) {
        CHECK_BOOL(CheckAttr(attrs, "bias", 1));

        std::vector<int> bias_shape = attrs.at("bias").shape;
        std::vector<char> bias      = attrs.at("bias").data;

        CHECK_BOOL(1 == bias_shape.size());

        bias_.resize(bias.size());

        bias_shape_[0] = bias_shape[0];

        // bias
        EigenTensorMap<float, 1> bias_original(
            reinterpret_cast<float*>(bias.data()),
            EigenDSize<1>(bias_shape[0]));
        EigenTensorMap<float, 1> bias_transform(
            reinterpret_cast<float*>(bias_.data()),
            bias_shape_);

        // copy only
        bias_transform = bias_original;
    }

    return Status::kSuccess;
}

Status Conv2d::InitWinograd() {
    if (3 == kernel_h_ && 3 == kernel_w_ && 1 == stride_h_ && 1 == stride_w_ &&
        1 == dilation_h_ && 1 == dilation_w_ && 1 == groups_ &&
        padding_t_ == padding_b_ && padding_t_ == padding_l_ &&
        padding_t_ == padding_r_ && (0 == padding_t_ || 1 == padding_t_)) {
        // convert weights
        int oc_up4               = (out_channels_ + 3) / 4 * 4;
        int weight_winograd_size = 16 * in_channels_ * oc_up4;

        weight_winograd_.resize(weight_winograd_size, 0.0f);

        float* src = (float*)weight_.data();
        float* dst = weight_winograd_.data();

        Conv3x3s1Winograd23TransformKernelPack4(src,
                                                in_channels_,
                                                out_channels_,
                                                dst);

        use_winograd_ = true;
    }

    return Status::kSuccess;
}

Status Conv2d::ForwardIm2Col(const Tensor& input, Tensor& output) {
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

    const EigenTensorMap<float, 4> kernel_tensor(
        reinterpret_cast<float*>(weight_.data()),
        weight_shape_);

    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    // reshape
    const int input_matrix_rows  = input_batch * output_height * output_width;
    const int input_matrix_cols  = kernel_h_ * kernel_w_ * input_channel;
    const int kernel_matrix_rows = kernel_h_ * kernel_w_ * input_channel;
    const int kernel_matrix_cols = output_channel;

    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 0)};
    EigenDSize<2> pre_contract_dims(input_matrix_rows, input_matrix_cols);
    EigenDSize<2> kernel_dims(kernel_matrix_rows, kernel_matrix_cols);
    EigenDSize<4> post_contract_dims(output_batch,
                                     output_height,
                                     output_width,
                                     output_channel);

    auto expr = input_eigen_tensor
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
                                           static_cast<float>(0))
                    .reshape(pre_contract_dims)
                    .contract(kernel_tensor.reshape(kernel_dims), contract_dims)
                    .reshape(post_contract_dims);

    if (use_bias_) {
        const EigenTensorMap<float, 1> bias(
            reinterpret_cast<float*>(bias_.data()),
            bias_shape_);

        output_eigen_tensor.device(*device) =
            expr + bias.reshape(EigenDSize<4>(1, 1, 1, bias_shape_[0]))
                       .broadcast(EigenDSize<4>(output_batch,
                                                output_height,
                                                output_width,
                                                1));

    } else {
        output_eigen_tensor.device(*device) = expr;
    }

    return Status::kSuccess;
}

Status Conv2d::ForwardIm2ColWithGroup(const Tensor& input, Tensor& output) {
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

    const EigenTensorMap<float, 4> kernel_tensor(
        reinterpret_cast<float*>(weight_.data()),
        weight_shape_);

    EigenTensorMap<float, 4> output_eigen_tensor =
        output.GetEigenTensor<float, 4>();

    // reshape
    const int input_channel_group  = input_channel / groups_;
    const int output_channel_group = output_channel / groups_;

    const int input_matrix_rows  = input_batch * output_height * output_width;
    const int input_matrix_cols  = kernel_h_ * kernel_w_ * input_channel_group;
    const int kernel_matrix_rows = kernel_h_ * kernel_w_ * input_channel_group;
    const int kernel_matrix_cols = output_channel_group;

    Eigen::array<Eigen::IndexPair<int>, 1> contract_dims = {
        Eigen::IndexPair<int>(1, 0)};
    EigenDSize<4> input_group_dims(input_batch,
                                   input_height,
                                   input_width,
                                   input_channel_group);
    EigenDSize<2> pre_contract_dims(input_matrix_rows, input_matrix_cols);
    EigenDSize<4> kernel_group_dims(kernel_h_,
                                    kernel_w_,
                                    input_channel_group,
                                    output_channel_group);
    EigenDSize<2> kernel_dims(kernel_matrix_rows, kernel_matrix_cols);
    EigenDSize<4> output_group_dims(output_batch,
                                    output_height,
                                    output_width,
                                    output_channel_group);

    for (int i = 0; i < groups_; ++i) {
        EigenDSize<4> input_start_index(0, 0, 0, i * input_channel_group);
        EigenDSize<4> kernel_start_index(0, 0, 0, i * output_channel_group);
        EigenDSize<4> output_start_index(0, 0, 0, i * output_channel_group);

        output_eigen_tensor.slice(output_start_index, output_group_dims)
            .device(*device) =
            input_eigen_tensor.slice(input_start_index, input_group_dims)
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
                                       static_cast<float>(0))
                .reshape(pre_contract_dims)
                .contract(
                    kernel_tensor.slice(kernel_start_index, kernel_group_dims)
                        .reshape(kernel_dims),
                    contract_dims)
                .reshape(output_group_dims);
    }

    if (use_bias_) {
        const EigenTensorMap<float, 1> bias(
            reinterpret_cast<float*>(bias_.data()),
            bias_shape_);

        output_eigen_tensor.device(*device) +=
            bias.reshape(EigenDSize<4>(1, 1, 1, bias_shape_[0]))
                .broadcast(EigenDSize<4>(output_batch,
                                         output_height,
                                         output_width,
                                         1));
    }

    return Status::kSuccess;
}

Status Conv2d::ForwardWinograd23(const Tensor& input, Tensor& output) {
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

    assert(input_batch == output_batch);

    const int tiles_h = (output_height + 1) / 2;
    const int tiles_w = (output_width + 1) / 2;

    const int output_channel_up4 = (output_channel + 3) / 4 * 4;
    const int weight_stride      = input_channel * output_channel_up4;

    const int M = tiles_h * tiles_w;
    const int N = output_channel;
    const int K = input_channel;

    const int batch       = input_batch;
    const int input_size  = input_height * input_width * input_channel;
    const int output_size = output_height * output_width * output_channel;
    const int output_spatial_size = output_height * output_width;

    const int input_buf_stride  = tiles_h * tiles_w * input_channel;
    const int output_buf_stride = tiles_h * tiles_w * output_channel;
    const int output_buf_stride_channel_up4 =
        tiles_h * tiles_w * output_channel_up4;

    // buffer
    if (input_buf_winograd_.size() < 16 * input_buf_stride) {
        input_buf_winograd_.resize(16 * input_buf_stride, 0.0f);
    }

    if (output_buf_winograd_.size() < 16 * output_buf_stride_channel_up4) {
        output_buf_winograd_.resize(16 * output_buf_stride_channel_up4, 0.0f);
    }

    const float* src  = input_eigen_tensor.data();
    float* src_buf    = input_buf_winograd_.data();
    float* weight_buf = weight_winograd_.data();
    float* dst_buf    = output_buf_winograd_.data();
    float* dst        = output_eigen_tensor.data();
    float* bias       = (float*)bias_.data();

    bool pad = (1 == padding_t_);

    for (int b = 0; b < batch; ++b) {
        Conv3x3s1Winograd23TransformInput(src + b * input_size,
                                          input_height,
                                          input_width,
                                          input_channel,
                                          pad,
                                          src_buf,
                                          input_buf_stride);

        SimpleInfer::Parallel(
            0,
            16,
            [&](size_t thread, size_t begin, size_t end) {
                for (size_t i = begin; i < end; ++i) {
                    GemmPack4F32(M,
                                 N,
                                 K,
                                 src_buf + i * input_buf_stride,
                                 input_channel,
                                 weight_buf + i * weight_stride,
                                 dst_buf + i * output_buf_stride,
                                 output_channel);
                }
            },
            16,
            1);

        Conv3x3s1Winograd23TransformOutput(dst_buf,
                                           output_buf_stride,
                                           dst + b * output_size,
                                           output_height,
                                           output_width,
                                           output_channel);

        if (use_bias_) {
            AddBiasNHWC(bias,
                        output_spatial_size,
                        output_channel,
                        dst + b * output_size);
        }
    }

    return Status::kSuccess;
}

}  // namespace SimpleInfer
