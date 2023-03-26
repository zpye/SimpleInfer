#ifndef SIMPLE_INFER_SRC_LAYER_HARD_SIGMOID_H_
#define SIMPLE_INFER_SRC_LAYER_HARD_SIGMOID_H_

#include "layer.h"

namespace SimpleInfer {

class HardSigmoid : public Layer {
public:
    HardSigmoid();

    virtual ~HardSigmoid() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    float alpha_ = 0.0f;
    float beta_  = 0.0f;
    float lower_ = 0.0f;
    float upper_ = 0.0f;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_HARD_SIGMOID_H_
