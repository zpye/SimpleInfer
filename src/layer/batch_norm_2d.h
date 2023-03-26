#ifndef SIMPLE_INFER_SRC_LAYER_BATCH_NORM_2D_H_
#define SIMPLE_INFER_SRC_LAYER_BATCH_NORM_2D_H_

#include "layer.h"

namespace SimpleInfer {

class BatchNorm2d : public Layer {
public:
    BatchNorm2d();

    virtual ~BatchNorm2d() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const Tensor& input, Tensor& output) override;

public:
    float eps_        = 1.000000e-05;
    int num_features_ = 0;

    EigenDSize<1> running_mean_shape_;
    std::vector<char> running_mean_;
    EigenDSize<1> running_var_shape_;
    std::vector<char> running_var_;

    bool use_affine_ = false;
    EigenDSize<1> weight_shape_;
    std::vector<char> weight_;
    EigenDSize<1> bias_shape_;
    std::vector<char> bias_;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_BATCH_NORM_2D_H_
