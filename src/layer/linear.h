#ifndef SIMPLE_INFER_SRC_LAYER_LINEAR_H_
#define SIMPLE_INFER_SRC_LAYER_LINEAR_H_

#include "layer.h"

namespace SimpleInfer {

class Linear : public Layer {
public:
    Linear();

    virtual ~Linear() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    int in_features_  = 0;
    int out_features_ = 0;

    bool use_bias_ = false;
    EigenDSize<2> weight_shape_;
    std::vector<char> weight_;
    EigenDSize<1> bias_shape_;
    std::vector<char> bias_;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_LINEAR_H_
