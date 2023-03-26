#ifndef SIMPLE_INFER_SRC_LAYER_FLATTEN_H_
#define SIMPLE_INFER_SRC_LAYER_FLATTEN_H_

#include "layer.h"

namespace SimpleInfer {

class Flatten : public Layer {
public:
    Flatten();

    virtual ~Flatten() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    int start_dim_ = 0;
    int end_dim_   = -1;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_FLATTEN_H_
