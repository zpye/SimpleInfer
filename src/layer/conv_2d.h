#ifndef SIMPLE_INFER_SRC_LAYER_CONV_2D_H_
#define SIMPLE_INFER_SRC_LAYER_CONV_2D_H_

#include "layer.h"

namespace SimpleInfer {

class Conv2d : public Layer {
public:
    Conv2d();

    virtual ~Conv2d() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Init(
        const std::map<std::string, pnnx::Parameter>& params,
        const std::map<std::string, pnnx::Attribute>& attrs) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    Status InitWeightAndBias(
        const std::map<std::string, pnnx::Parameter>& params,
        const std::map<std::string, pnnx::Attribute>& attrs);

    Status InitWinograd();

    Status ForwardIm2Col(const Tensor& input, Tensor& output);

    Status ForwardIm2ColWithGroup(const Tensor& input, Tensor& output);

    Status ForwardWinograd23(const Tensor& input, Tensor& output);

public:
    enum class PaddingMode { kZeros = 0, kReplicate, kReflect } padding_mode_;
    int padding_t_    = 0;
    int padding_b_    = 0;
    int padding_l_    = 0;
    int padding_r_    = 0;
    int kernel_h_     = 0;
    int kernel_w_     = 0;
    int stride_h_     = 0;
    int stride_w_     = 0;
    int dilation_h_   = 0;
    int dilation_w_   = 0;
    int groups_       = 0;
    int in_channels_  = 0;
    int out_channels_ = 0;

    bool use_bias_ = false;
    EigenDSize<4> weight_shape_;
    std::vector<char> weight_;
    EigenDSize<1> bias_shape_;
    std::vector<char> bias_;

    // winograd
    bool use_winograd_ = false;
    int tiles_h_       = 0;
    int tiles_w_       = 0;

    std::vector<float> weight_winograd_;
    std::vector<float> input_buf_winograd_;
    std::vector<float> output_buf_winograd_;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_CONV_2D_H_
