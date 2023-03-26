#ifndef SIMPLE_INFER_SRC_LAYER_MAX_POOL_2D_H_
#define SIMPLE_INFER_SRC_LAYER_MAX_POOL_2D_H_

#include "layer.h"

namespace SimpleInfer {

class MaxPool2d : public Layer {
public:
    MaxPool2d();

    virtual ~MaxPool2d() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    bool ceil_mode_      = false;
    bool return_indices_ = false;
    int padding_t_       = 0;
    int padding_b_       = 0;
    int padding_l_       = 0;
    int padding_r_       = 0;
    int kernel_h_        = 0;
    int kernel_w_        = 0;
    int stride_h_        = 0;
    int stride_w_        = 0;
    int dilation_h_      = 0;
    int dilation_w_      = 0;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_MAX_POOL_2D_H_
