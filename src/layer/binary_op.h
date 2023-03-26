#ifndef SIMPLE_INFER_SRC_LAYER_BINARY_OP_H_
#define SIMPLE_INFER_SRC_LAYER_BINARY_OP_H_

#include "layer.h"

namespace SimpleInfer {

class BinaryOp : public Layer {
public:
    BinaryOp();

    virtual ~BinaryOp() override;

public:
    virtual Status Init(const pnnx::Operator* op) override;

    virtual Status Validate() override;

    virtual Status Forward(const std::vector<Tensor>& inputs,
                           Tensor& output) override;

public:
    Status BroadcastShape(const std::vector<int>& shape0,
                          const std::vector<int>& shape1,
                          std::vector<int>& broadcast_shape);

public:
    enum class BinaryOpType { kAdd = 0, kMul = 2 } binary_op_type_;
};

}  // namespace SimpleInfer

#endif  // SIMPLE_INFER_SRC_LAYER_BINARY_OP_H_
