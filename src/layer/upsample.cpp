#include "upsample.h"

#include <cmath>

#if defined(USE_HALIDE)
#include "HalideBuffer.h"
#include "halide_upsample_nearest.h"
#endif

namespace SimpleInfer {

DEFINE_LAYER_REGISTRY(Upsample);

Upsample::Upsample() {}

Upsample::~Upsample() {}

Status Upsample::Init(const pnnx::Operator* op) {
    Status ret = Layer::Init(op);
    if (Status::kSuccess != ret) {
        return ret;
    }

    CHECK_BOOL(CheckParam(op, "mode", 4));
    if ("nearest" == op->params.at("mode").s) {
        upsample_mode_ = UpsampleMode::kNearest;
    } else {
        LOG(ERROR) << "Upsample::Init fail ["
                   << "unsupport upsample mode"
                   << "]";
        return Status::kUnsupport;
    }

    CHECK_BOOL(CheckParam(op, "scale_factor", 6) || CheckParam(op, "size", 5));

    if (CheckParam(op, "scale_factor", 6)) {
        CHECK_BOOL(2 == op->params.at("scale_factor").af.size());
        scale_factor_h_ = op->params.at("scale_factor").af[0];
        scale_factor_w_ = op->params.at("scale_factor").af[1];
    }

    // TODO: size

    return Status::kSuccess;
}

Status Upsample::Validate() {
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
        LOG(ERROR) << "Upsample::Validate fail ["
                   << "unsupport input/output data type"
                   << "]";
        return Status::kUnsupport;
    }

    // TODO: check shape

    return Status::kSuccess;
}

struct Nearest4D {
    Nearest4D(const EigenTensorMap<float, 4>& _input,
              float _scale_h_inv,
              float _scale_w_inv)
        : input(_input),
          input_dsize(_input.dimensions()),
          scale_h_inv(_scale_h_inv),
          scale_w_inv(_scale_w_inv) {}

    float operator()(const Eigen::array<Eigen::DenseIndex, 4>& coord) const {
        int h_in = (float)coord[1] * scale_h_inv;
        h_in     = (std::max)(0, (std::min)((int)(input_dsize[1] - 1), h_in));

        int w_in = (float)coord[2] * scale_w_inv;
        w_in     = (std::max)(0, (std::min)((int)(input_dsize[2] - 1), w_in));

        return input(coord[0], h_in, w_in, coord[3]);
    }

    const EigenTensorMap<float, 4>& input;
    const EigenDSize<4>& input_dsize;
    const float scale_h_inv;
    const float scale_w_inv;
};

Status Upsample::Forward(const Tensor& input, Tensor& output) {
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

    const float scale_factor_h_inv = 1.0f / scale_factor_h_;
    const float scale_factor_w_inv = 1.0f / scale_factor_w_;

    // nearest
#if defined(USE_HALIDE)
    Halide::Runtime::Buffer<float> input_buffer(
        input_eigen_tensor.data(),
        {input_channel, input_width, input_height, input_batch});

    Halide::Runtime::Buffer<float> output_buffer(
        output_eigen_tensor.data(),
        {output_channel, output_width, output_height, output_batch});

    int ret = halide_upsample_nearest(scale_factor_w_inv,
                                      scale_factor_h_inv,
                                      input_buffer,
                                      output_buffer);
    if (0 != ret) {
        LOG(ERROR) << "error in halide_upsample_nearest: " << ret;
        return Status::kFail;
    }

#else
#if 0
    for (int h = 0; h < output_height; ++h) {
        int h_in = (float)h * scale_factor_h_inv;
        h_in     = (std::max)(0, (std::min)(input_height - 1, h_in));

        for (int w = 0; w < output_width; ++w) {
            int w_in = (float)w * scale_factor_w_inv;
            w_in     = (std::max)(0, (std::min)(input_width - 1, w_in));

            output_eigen_tensor
                .slice(EigenDSize<4>(0, h, w, 0),
                       EigenDSize<4>(input_batch, 1, 1, input_channel))
                .device(*device) = input_eigen_tensor.slice(
                EigenDSize<4>(0, h_in, w_in, 0),
                EigenDSize<4>(input_batch, 1, 1, input_channel));
        }
    }
#else
    output_eigen_tensor.device(*device) = output_eigen_tensor.generate(
        Nearest4D(input_eigen_tensor, scale_factor_h_inv, scale_factor_w_inv));
#endif
#endif  // USE_HALIDE

    return Status::kSuccess;
}

}  // namespace SimpleInfer
