#ifndef SIMPLE_INFER_SRC_LAYER_CAT_H_
#define SIMPLE_INFER_SRC_LAYER_CAT_H_

#include "layer.h"

namespace SimpleInfer {

class Cat : public Layer {
public:
    Cat();

    virtual ~Cat() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const std::vector<Tensor>& inputs,
                           Tensor& output) override;

public:
    int dim_ = 0;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_CAT_H_
