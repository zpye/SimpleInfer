#ifndef SIMPLE_INFER_SRC_LAYER_HARD_SWISH_H_
#define SIMPLE_INFER_SRC_LAYER_HARD_SWISH_H_

#include "layer.h"

namespace SimpleInfer {

class HardSwish : public Layer {
public:
    HardSwish();

    virtual ~HardSwish() override;

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

#endif  // SIMPLE_INFER_SRC_LAYER_HARD_SWISH_H_
