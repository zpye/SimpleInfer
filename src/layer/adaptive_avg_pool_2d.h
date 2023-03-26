#ifndef SIMPLE_INFER_SRC_LAYER_ADAPTIVE_AVG_POOL_2D_H_
#define SIMPLE_INFER_SRC_LAYER_ADAPTIVE_AVG_POOL_2D_H_

#include "layer.h"

namespace SimpleInfer {

class AdaptiveAvgPool2d : public Layer {
public:
    AdaptiveAvgPool2d();

    virtual ~AdaptiveAvgPool2d() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    int output_h_ = 0;
    int output_w_ = 0;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_ADAPTIVE_AVG_POOL_2D_H_
