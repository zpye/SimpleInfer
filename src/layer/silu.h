#ifndef SIMPLE_INFER_SRC_LAYER_SILU_H_
#define SIMPLE_INFER_SRC_LAYER_SILU_H_

#include "layer.h"

namespace SimpleInfer {

class SiLU : public Layer {
public:
    SiLU();

    virtual ~SiLU() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_SILU_H_
