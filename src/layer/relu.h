#ifndef SIMPLE_INFER_SRC_LAYER_RELU_H_
#define SIMPLE_INFER_SRC_LAYER_RELU_H_

#include "layer.h"

namespace SimpleInfer {

class ReLU : public Layer {
public:
    ReLU();

    virtual ~ReLU() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_RELU_H_
