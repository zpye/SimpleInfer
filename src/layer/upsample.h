#ifndef SIMPLE_INFER_SRC_LAYER_UPSAMPLE_H_
#define SIMPLE_INFER_SRC_LAYER_UPSAMPLE_H_

#include "layer.h"

namespace SimpleInfer {

class Upsample : public Layer {
public:
    Upsample();

    virtual ~Upsample() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    enum class UpsampleMode { kNearest = 0 } upsample_mode_;
    float scale_factor_h_ = 0.0f;
    float scale_factor_w_ = 0.0f;
    int size_h_           = 0;
    int size_w_           = 0;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_UPSAMPLE_H_
